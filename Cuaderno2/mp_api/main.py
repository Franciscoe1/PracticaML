"""
API REST con FastAPI para la Neurona de McCulloch y Pitts (MP)
aplicada al dataset Breast Cancer Wisconsin.

Ejecutar:  uvicorn main:app --reload
Docs:      http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

N_FEATURES = 30


# ---------------------------------------------------------------------------
# Modelo (misma lógica del notebook, pero vectorizada con numpy)
# ---------------------------------------------------------------------------
class MPNeuron:
    def __init__(self):
        self.threshold = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        # z = suma de entradas binarias; se activa si z >= threshold
        return X.sum(axis=1) >= self.threshold

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            accuracy[th] = accuracy_score(self.predict(X), Y)
        self.threshold = max(accuracy, key=accuracy.get)


class Binarizer:
    """
    Equivale a X.apply(pd.cut, bins=2, labels=[1, 0]) del notebook, pero con
    puntos de corte FIJOS calculados con los datos de entrenamiento.
    (En una API no se puede recalcular el corte con cada petición.)
    Valor <= punto medio del rango -> 1, valor > punto medio -> 0.
    """

    def fit(self, X: np.ndarray) -> "Binarizer":
        self.cuts = (X.min(axis=0) + X.max(axis=0)) / 2
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X <= self.cuts).astype(int)


# ---------------------------------------------------------------------------
# Entrenamiento al arrancar la API
# ---------------------------------------------------------------------------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, stratify=data.target, random_state=42
    )
    binarizer = Binarizer().fit(X_train)
    neuron = MPNeuron()
    neuron.fit(binarizer.transform(X_train), y_train)

    acc = accuracy_score(y_test, neuron.predict(binarizer.transform(X_test)))
    state.update(
        binarizer=binarizer,
        neuron=neuron,
        feature_names=list(data.feature_names),
        target_names=list(data.target_names),  # ['malignant', 'benign']
        test_accuracy=float(acc),
    )
    yield
    state.clear()


app = FastAPI(
    title="API Neurona MP - Breast Cancer",
    description="Clasificador binario con una neurona de McCulloch y Pitts. "
    "Proyecto académico, NO apto para diagnóstico médico.",
    version="1.0.0",
    lifespan=lifespan,
)


# Permite llamar a la API desde un index.html abierto como archivo local
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "index.html")


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description="30 características, en el orden de GET /features",
    )


class BatchRequest(BaseModel):
    samples: list[list[float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    prediction: int = Field(description="0 = malignant, 1 = benign")
    label: str
    neuron_sum: int = Field(description="Suma de entradas binarias activas")
    threshold: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _predict(X: np.ndarray) -> list[PredictResponse]:
    binarizer, neuron = state["binarizer"], state["neuron"]
    Xb = binarizer.transform(X)
    preds = neuron.predict(Xb).astype(int)
    sums = Xb.sum(axis=1)
    return [
        PredictResponse(
            prediction=int(p),
            label=state["target_names"][int(p)],
            neuron_sum=int(s),
            threshold=int(neuron.threshold),
        )
        for p, s in zip(preds, sums)
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": "McCulloch-Pitts Neuron",
        "threshold": int(state["neuron"].threshold),
        "n_features": N_FEATURES,
        "test_accuracy": state["test_accuracy"],
        "classes": {0: state["target_names"][0], 1: state["target_names"][1]},
    }


@app.get("/features")
def features():
    return {"features": state["feature_names"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _predict(np.array([req.features], dtype=float))[0]


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchRequest):
    if any(len(s) != N_FEATURES for s in req.samples):
        raise HTTPException(
            status_code=422,
            detail=f"Cada muestra debe tener exactamente {N_FEATURES} valores.",
        )
    return _predict(np.array(req.samples, dtype=float))
