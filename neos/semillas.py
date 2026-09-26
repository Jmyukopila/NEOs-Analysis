"""Semillas globales y determinismo del proyecto.

El repositorio tenía tres semillas distintas (20 en el notebook de ML, 42 en el
de regresión y en los scripts) y ninguna forma de cuantificar la incertidumbre de
una métrica. Este módulo centraliza las dos cosas:

1. `RANDOM_STATE`, la semilla canónica, y `SEMILLAS`, la lista con la que se
   repite cualquier experimento.
2. `fijar_semillas_todas()`, que fija el generador de numpy, el de Python y el
   número de hilos, de modo que dos ejecuciones del mismo script produzcan los
   mismos bytes.
"""

import os
import random

import numpy as np

from neos.constantes import N_SEMILLAS, RANDOM_STATE, SEMILLAS

__all__ = [
    "RANDOM_STATE",
    "SEMILLAS",
    "N_SEMILLAS",
    "fijar_semilla",
    "fijar_semillas_todas",
    "generar_semillas",
    "cv_de_semillas",
]

# Semilla global derivada: cada componente estocástico propio (subsampling de
# XGBoost, split inicial de K-Means) recibe una semilla distinta pero
# determinista, para que repetir el pipeline no cambie el resultado.
_OFFSET = {"sklearn": 1000, "xgboost": 2000, "kmeans": 3000, "shap": 4000}


def fijar_semilla(semilla=RANDOM_STATE, componentes=("sklearn", "xgboost", "kmeans", "shap")):
    """Fija una semilla y devuelve el dict de semillas derivadas por componente.

    Devolver el dict (y no solo aplicarlo) permite escribirlo en el JSON de
    resultados: un número que no se puede reconstruir no es reproducible.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    derivadas = {"global": semilla}
    derivadas.update({c: semilla + _OFFSET[c] for c in componentes})
    return derivadas


def fijar_semillas_todas(semilla=RANDOM_STATE, n_threads=1):
    """Fija Python, numpy y el número de hilos de BLAS/OpenMP.

    El número de hilos importa de verdad aquí: XGBoost y las álgebra de sklearn
    reducen en paralelo, y esa reducción es no determinista, así que dos
    ejecuciones con distinto `n_threads` pueden dar métricas distintas en el
    último decimal.
    """
    derivadas = fijar_semilla(semilla)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n_threads)
    return derivadas


def generar_semillas(n=N_SEMILLAS, base=RANDOM_STATE):
    """Las `n` semillas del experimento, derivadas de la semilla canónica.

    Deliberadamente contiguas y no aleatorias: una lista de semillas que cambia
    entre ejecuciones hace incomparables dos resultados.
    """
    return tuple(base + i for i in range(n))


def cv_de_semillas(semilla, n_splits=5, n_repeats=1, estratificar=True):
    """Un splitter de CV con la semilla dada, para repetir el experimento.

    Se usa en lugar de `crear_cv()` cuando lo que se quiere no es un estimador
    puntual sino la distribución del estimador sobre repartos distintos.
    """
    from sklearn.model_selection import KFold, RepeatedStratifiedKFold

    if estratificar:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=semilla)
    return KFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
