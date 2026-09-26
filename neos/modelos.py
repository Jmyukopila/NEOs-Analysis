"""Modelos, ponderación de desbalance y métricas de la clasificación supervisada.

Tres correcciones respecto a la versión que había en el repo:

*   `peso_positivos()` ya no puede devolver un valor menor que 1. Antes
    `n_neg/n_pos` con la mayoría positiva (PHA 5.9 % en la población completa, pero
    mayoritaria en algunas subpoblaciones, p. ej. tras 2015) devolvía 0.4, y
    `scale_pos_weight=0.4` hace que XGBoost penalice a la clase positiva. El factor
    correcto es `n_neg/n_pos` cuando la positiva es minoritaria, y su inverso cuando
    no.
*   `SimpleImputer` se ajusta dentro de cada fold. `matriz_features()` lo hacía una
    sola vez sobre todos los datos, incluida la parte que se usa como prueba.
*   `class_weight="balanced"` se sustituye por el factor explícito en la regresión
    logística, porque "balanced" usa el desbalance de las clases que ve el estimador,
    y dentro de un fold eso no es el desbalance de la población: hace el resultado
    dependiente del reparto de folds.
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from neos.constantes import RANDOM_STATE

__all__ = [
    "SCORING",
    "peso_positivos",
    "crear_xgb",
    "crear_logreg",
    "crear_random_forest",
    "crear_modelos",
    "crear_cv",
    "matriz_features",
    "con_imputador",
    "umbral_por_f2",
]

#: Métrica de selección. F2 porque el coste de no detectar un PHA es mayor que el
#: de un falso positivo: un PHA sin marcar pasa por la vía lenta de la NASA.
SCORING = {"F2": make_scorer(fbeta_score, beta=2),
           "PR-AUC": "average_precision",
           "ROC-AUC": "roc_auc"}

#: β de la F-beta que se optimiza. Explícito y no `beta=2` en tres sitios distintos.
BETA = 2


def peso_positivos(y):
    """Factor de compensación del desbalance, siempre >= 1.

    `n_neg/n_pos` si la clase positiva es la minoritaria; `n_pos/n_neg` si es la
    mayoritaria. Devolver un factor < 1 no compensa el desbalance: lo invierte, y en
    XGBoost eso se traduce en un modelo que penaliza a la clase minoritaria y
    predice la mayoritaria con una precisión aparente alta.
    """
    y = np.asarray(y)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("peso_positivos() necesita ambas clases presentes.")
    return n_neg / n_pos if n_pos <= n_neg else n_pos / n_neg


def crear_xgb(y, random_state=RANDOM_STATE):
    """XGBoost del proyecto, con `scale_pos_weight` ajustado a la `y` que reciba."""
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                         random_state=random_state, eval_metric="logloss",
                         scale_pos_weight=peso_positivos(y), n_jobs=1)


def crear_logreg(y, random_state=RANDOM_STATE):
    """Regresión logística con el factor explícito de desbalance.

    `class_weight` como dict {0: 1, 1: peso} y no `"balanced"`: la razón es que
    "balanced" recalcula el factor con el desbalance del fold, de modo que el mismo
    modelo da resultados distintos según cómo se repartan los folds.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                   class_weight={0: 1.0, 1: peso_positivos(y)},
                                   random_state=random_state)),
    ])


def crear_random_forest(y=None, random_state=RANDOM_STATE):
    """Bosque aleatorio. `class_weight='balanced_subsample'` pondera cada fold por
    su propio desbalance; se deja así a propósito y se documenta como tal, porque
    para RF el desbalance por fold es un ruido aceptable y recalcularlo por fold
    multiplica el coste de ajuste sin ganancia medida."""
    return RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  random_state=random_state, n_jobs=1)


def crear_modelos(y, random_state=RANDOM_STATE, con_imputacion=True):
    """Los tres clasificadores comparados, con imputación dentro del pipeline.

    XGBoost y el bosque toleran NaN de forma nativa, pero la regresión logística no.
    Anteponer el imputador a los tres mantiene idéntico el preprocesamiento, que es
    lo que hace comparables las columnas de importancia y las curvas ROC.
    """
    modelos = {
        "LogReg": crear_logreg(y, random_state=random_state),
        "RandForest": crear_random_forest(y, random_state=random_state),
        "XGBoost": crear_xgb(y, random_state=random_state),
    }
    if con_imputacion:
        return {k: con_imputador(v) for k, v in modelos.items()}
    return modelos


def crear_cv(n_splits=5, n_repeats=1, random_state=RANDOM_STATE):
    """CV estratificado. `n_repeats=1` por defecto: el nº de repeticiones lo decide
    `generar_semillas()`, y tener también las repeticiones dentro del CV confunde
    "variación entre folds" con "variación entre semillas"."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)


def con_imputador(estimador, estrategia="median"):
    """Antepone un `SimpleImputer` ajustado dentro del fold, no antes."""
    return Pipeline([("imputer", SimpleImputer(strategy=estrategia)), ("clf", estimador)])


def matriz_features(obj, cols):
    """Features a nivel objeto imputadas por mediana.

    Se mantiene por compatibilidad con los notebooks, pero el pipeline ya no la usa:
    imputar aquí significa usar la mediana de la parte de prueba, que es fuga.
    """
    return obj[cols].fillna(obj[cols].median()).to_numpy(dtype=float)


def umbral_por_f2(y_true, y_score, beta=BETA, n_cortes=201):
    """Umbral de probabilidad que maximiza F-beta, y el valor de la métrica.

    Se calcula sobre las predicciones de validación, nunca sobre las de prueba: un
    umbral elegido con la prueba convierte la prueba en entrenamiento y la métrica
    que se reporta deja de ser una estimación de nada.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    cortes = np.linspace(y_score.min(), y_score.max(), n_cortes)
    valores = [fbeta_score(y_true, (y_score >= c).astype(int), beta=beta, zero_division=0)
               for c in cortes]
    mejor = int(np.argmax(valores))
    return {"umbral": float(cortes[mejor]), "F2": float(valores[mejor])}


def ajustar_y_calibrar(modelo, X, y, semilla=RANDOM_STATE):
    """Ajusta el modelo y devuelve el umbral F2 del fold de validación.

    Uso típico dentro de un bucle de CV: se pasa el fold de entrenamiento dividido en
    `entren`/`val`, se elige el umbral en `val` y se aplica a `test`. Así el umbral y
    el test siguen siendo independientes.
    """
    from sklearn.model_selection import train_test_split

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.25,
                                            random_state=semilla, stratify=y)
    modelo = clone(modelo)
    modelo.fit(Xtr, ytr)
    umbral = umbral_por_f2(yval, modelo.predict_proba(Xval)[:, 1])["umbral"]
    return modelo, umbral


def f2_en_corte(y_true, y_score, umbral, beta=BETA):
    return fbeta_score(y_true, (np.asarray(y_score) >= umbral).astype(int),
                       beta=beta, zero_division=0)


def recall_en_corte(y_true, y_score, umbral):
    return recall_score(y_true, (np.asarray(y_score) >= umbral).astype(int),
                        zero_division=0)
