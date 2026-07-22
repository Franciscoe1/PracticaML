# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluacion(y_true, y_pred, metricas):
    """
    Evalúa un modelo utilizando las métricas proporcionadas.

    Parámetros
    ----------
    y_true : valores reales
    y_pred : predicciones
    metricas : diccionario
        {'ACC': funcion, 'PREC': funcion, ...}

    Retorna
    -------
    DataFrame con los resultados.
    """
    resultados = {}

    for nombre, metrica in metricas.items():
        resultados[nombre] = metrica(y_true, y_pred)

    return pd.DataFrame(resultados, index=["Resultado"])


def mapa_modelo_clasif_2d(X, y, modelo, resultados=None, nombre="Modelo"):
    """
    Dibuja las fronteras de decisión para un clasificador de dos variables.
    """

    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
    plt.scatter(
        X[:,0],
        X[:,1],
        c=y,
        edgecolors='k',
        cmap=plt.cm.Set1,
        s=45
    )

    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.title(f"Frontera de decisión - {nombre}")

    if resultados is not None:
        texto = "\n".join(
            [f"{col}: {resultados.iloc[0][col]:.4f}" for col in resultados.columns]
        )

        plt.text(
            0.02,
            0.98,
            texto,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    plt.grid(True)
    plt.show()
