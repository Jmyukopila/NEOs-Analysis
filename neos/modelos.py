"""Modelos, ponderación de desbalance y métricas de la clasificación supervisada."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from neos.constantes import RANDOM_STATE

SCORING = {"F2": make_scorer(fbeta_score, beta=2),
           "PR-AUC": "average_precision",
           "ROC-AUC": "roc_auc"}


def peso_positivos(y):
    """nº negativos / nº positivos: factor de compensación del desbalance (~13:1)."""
    return (y == 0).sum() / max((y == 1).sum(), 1)


def crear_xgb(y, random_state=RANDOM_STATE):
    """XGBoost del proyecto, con scale_pos_weight ajustado a la `y` que reciba."""
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                         random_state=random_state, eval_metric="logloss",
                         scale_pos_weight=peso_positivos(y), n_jobs=1)


def crear_modelos(y, random_state=RANDOM_STATE):
    """Los tres clasificadores comparados. XGBoost pondera con scale_pos_weight;
    LogReg y RandomForest con class_weight."""
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced",
                                                   random_state=random_state)),
        "RandForest": crear_random_forest(random_state=random_state),
        "XGBoost": crear_xgb(y, random_state=random_state),
    }


def crear_random_forest(random_state=RANDOM_STATE):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=random_state, n_jobs=1)


def crear_cv(n_splits=5, n_repeats=3, random_state=RANDOM_STATE):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)


def matriz_features(obj, cols):
    """Features a nivel objeto con los pocos NaN residuales imputados por mediana."""
    return obj[cols].fillna(obj[cols].median()).values
