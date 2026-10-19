"""Entrena un modelo de Regresión Logística con solo 10 características.
Ejecutar primero: python entrenar_modelo.py
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
COLUMNAS = [
    "worst concave points", "worst perimeter", "mean concave points",
    "worst radius", "mean perimeter", "worst area", "mean radius",
    "mean area", "mean concavity", "worst concavity"
]

datos = load_breast_cancer(as_frame=True)
X = datos.data[COLUMNAS].copy()
# En el dataset original: 0 = maligno, 1 = benigno.
# Se cambia la codificación a 0 = benigno, 1 = maligno.
y = (datos.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
modelo = Pipeline([
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))
])
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

joblib.dump(modelo, CARPETA / "modelo_cancer.pkl")
joblib.dump(COLUMNAS, CARPETA / "columnas_modelo.pkl")

print(f"Exactitud en datos de prueba: {accuracy_score(y_test, predicciones) * 100:.2f}%")
print("\nMatriz de confusión (0=benigno, 1=maligno):")
print(confusion_matrix(y_test, predicciones, labels=[0, 1]))
print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones, labels=[0, 1],
                            target_names=["Benigno", "Maligno"], zero_division=0))
print("\nModelo guardado en:", CARPETA / "modelo_cancer.pkl")
print("Columnas guardadas en:", CARPETA / "columnas_modelo.pkl")
print("Total de características:", len(COLUMNAS))
