"""Entrena un modelo de Regresión Logística para el conjunto Wisconsin de cáncer de mama.
Ejecutar antes de iniciar la API: python entrenar_modelo.py
Proyecto académico; no sirve para diagnosticar pacientes reales.
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CARPETA = Path(__file__).resolve().parent
datos = load_breast_cancer(as_frame=True)
X, y = datos.data, datos.target  # 0 = maligno, 1 = benigno
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
modelo = Pipeline([
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))
])
modelo.fit(X_entrenamiento, y_entrenamiento)
predicciones = modelo.predict(X_prueba)
joblib.dump(modelo, CARPETA / "modelo_cancer.pkl")
joblib.dump(list(X.columns), CARPETA / "columnas_modelo.pkl")
print(f"Exactitud en prueba: {accuracy_score(y_prueba, predicciones) * 100:.2f}%")
print("Matriz de confusión:")
print(confusion_matrix(y_prueba, predicciones))
print("Reporte de clasificación:")
print(classification_report(y_prueba, predicciones, target_names=["Maligno", "Benigno"]))
print("Archivos guardados en:", CARPETA)
