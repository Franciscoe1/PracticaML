from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CARPETA = Path(__file__).resolve().parent
modelo = joblib.load(CARPETA / "modelo_cancer.pkl")
columnas = joblib.load(CARPETA / "columnas_modelo.pkl")
app = FastAPI(title="API de predicción de cáncer de mama - Regresión Logística",
              description="Prototipo académico. No constituye un diagnóstico médico.", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

class DatosPaciente(BaseModel):
    parametros: dict[str, float] = Field(..., description="Las 30 características numéricas del conjunto Wisconsin")

@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente", "modelo": "Regresión Logística", "total_parametros": len(columnas)}

@app.get("/parametros")
def obtener_parametros():
    return {"parametros": columnas}

@app.get("/registros-reales")
def registros_reales():
    from sklearn.datasets import load_breast_cancer
    datos = load_breast_cancer(as_frame=True)
    df = datos.data.copy()
    df["diagnostico_real"] = ["Maligno" if valor == 0 else "Benigno" for valor in datos.target]
    return {"total": len(df), "registros": df.head(20).to_dict(orient="records")}

@app.post("/predecir")
def predecir(datos: DatosPaciente):
    faltantes = sorted(set(columnas) - set(datos.parametros))
    if faltantes:
        raise HTTPException(status_code=422, detail={"mensaje": "Faltan características", "parametros_faltantes": faltantes})
    try:
        fila = pd.DataFrame([[float(datos.parametros[c]) for c in columnas]], columns=columnas)
        prediccion = int(modelo.predict(fila)[0])
        probabilidades = modelo.predict_proba(fila)[0]
        nombres = {0: "Maligno", 1: "Benigno"}
        estimadas = {nombres[int(clase)]: float(prob) for clase, prob in zip(modelo.classes_, probabilidades)}
        return {"prediccion": prediccion, "resultado": nombres[prediccion],
                "probabilidades_estimadas": estimadas, "modelo": "Regresión Logística",
                "advertencia": "Resultado experimental académico; no constituye un diagnóstico médico."}
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"No se pudieron procesar los datos: {error}")
