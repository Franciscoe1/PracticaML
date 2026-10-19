"""API FastAPI para el modelo entrenado con 10 características.
Ejecutar: python -m uvicorn api:app --reload
Proyecto académico; no constituye un diagnóstico médico.
"""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CARPETA = Path(__file__).resolve().parent
if not (CARPETA / "modelo_cancer.pkl").exists() or not (CARPETA / "columnas_modelo.pkl").exists():
    raise RuntimeError("Faltan los archivos del modelo. Ejecuta primero: python entrenar_modelo.py")

modelo = joblib.load(CARPETA / "modelo_cancer.pkl")
columnas = joblib.load(CARPETA / "columnas_modelo.pkl")

app = FastAPI(
    title="API académica de clasificación de cáncer de mama",
    description="Regresión Logística entrenada con 10 características. No es un diagnóstico médico.",
    version="4.0.0"
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"]
)

class DatosPaciente(BaseModel):
    parametros: dict[str, float] = Field(..., description="Las 10 características seleccionadas")

@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente", "modelo": "Regresión Logística",
            "total_parametros": len(columnas), "caracteristicas": columnas}

@app.get("/parametros")
def obtener_parametros():
    return {"parametros": columnas}

@app.post("/predecir")
def predecir(datos: DatosPaciente):
    faltantes = sorted(set(columnas) - set(datos.parametros))
    if faltantes:
        raise HTTPException(status_code=422, detail={
            "mensaje": "Faltan características", "parametros_faltantes": faltantes
        })
    try:
        fila = pd.DataFrame([[float(datos.parametros[c]) for c in columnas]], columns=columnas)
        prediccion = int(modelo.predict(fila)[0])
        probabilidades = modelo.predict_proba(fila)[0]
        estimadas = {
            ("Maligno" if int(clase) == 0 else "Benigno"): float(prob)
            for clase, prob in zip(modelo.classes_, probabilidades)
        }
        maligno = prediccion == 0
        return {
            "prediccion": prediccion,
            "resultado": "Maligno" if maligno else "Benigno",
            "clasificacion": "MALIGNA" if maligno else "BENIGNA",
            "mensaje": "El paciente tiene cáncer" if maligno else "El paciente no tiene cáncer",
            "probabilidades_estimadas": estimadas,
            "modelo": "Regresión Logística con 10 características",
            "advertencia": "Resultado experimental académico; no constituye un diagnóstico médico."
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"No se pudieron procesar los datos: {error}")
