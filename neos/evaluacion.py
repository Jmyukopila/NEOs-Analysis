"""Métricas, intervalos de confianza y contraste entre clasificadores.

Lo que faltaba en el repositorio era cualquier medida de incertidumbre. Todas las
cifras se reportaban como un número único, lo que hace indistinguible "este modelo
es mejor" de "esta semilla dio una vuelta buena". Este módulo añade:

*   `intervalo_bootstrap()`: IC 95 % por re-muestreo de la muestra de prueba.
*   `mcnemar()`: contraste apareado para dos clasificadores sobre la misma muestra,
    que es el test adecuado aquí porque las predicciones no son independientes.
*   `linea_base_ap()`: la precisión de la línea base, para que PR-AUC tenga
    referencia. Sin esto, un 0.30 de average precision no dice nada: si la clase
    positiva es el 2 %, el azar ya da 0.02 y un modelo que no aprende nada da
    bastante más por la forma de la curva.
*   `resumen_holdout()`: la tabla que va a `results/tables/`, con etiqueta, umbral y
    n en cada fila para que sea interpretable sin contexto.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neos.modelos import BETA

__all__ = [
    "resumen_holdout",
    "intervalo_bootstrap",
    "mcnemar",
    "linea_base_ap",
    "tabla_comparativa",
    "media_entre_semillas",
]


def _metricas(y_true, y_score, umbral, beta=BETA):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= umbral).astype(int)
    m = {}
    if 0 < y_true.sum() < len(y_true):  # roc_auc no está definida con una sola clase
        m["ROC-AUC"] = roc_auc_score(y_true, y_score)
    m["PR-AUC"] = average_precision_score(y_true, y_score)
    m["Brier"] = brier_score_loss(y_true, y_score)
    m["F2"] = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    m["precision"] = precision_score(y_true, y_pred, zero_division=0)
    m["recall"] = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m.update({"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)})
    m["n"] = int(len(y_true))
    m["n_pos"] = int(y_true.sum())
    m["umbral"] = float(umbral)
    return m


def resumen_holdout(y_true, y_score, umbral, etiqueta="et_pha", beta=BETA):
    """Métricas de un holdout, con la etiqueta declarada en el propio resultado.

    Devolver la etiqueta dentro del dict (y no solo como título de figura) es lo que
    permite que `tabla_comparativa()` pueda juntar filas de etiquetas distintas sin
    que el lector tenga que saber de dónde salió cada una.
    """
    m = _metricas(y_true, y_score, umbral, beta=beta)
    m["etiqueta"] = etiqueta
    return m


def intervalo_bootstrap(y_true, y_score, umbral, metrica="F2", n_boot=2000,
                        alpha=0.05, semilla=20, beta=BETA):
    """IC percentil por re-muestreo con reemplazo de las filas de prueba.

    Re-muestrea *pares* (y_true, y_pred), no las etiquetas por separado: la
    correlación entre ambas es justamente lo que se quiere conservar.

    Es un IC condicional al conjunto de prueba. No cubre la incertidumbre del
    ajuste del modelo, que es de otro orden y se mide repitiendo el CV con distintas
    semillas (`media_entre_semillas`). Decirlo evita el error habitual de presentar
    este intervalo como si fuera todo el error posible.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= umbral).astype(int)
    rng = np.random.default_rng(semilla)
    n = len(y_true)
    valores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if yt.sum() in (0, len(yt)):
            valores[b] = np.nan
            continue
        if metrica == "F2":
            valores[b] = fbeta_score(yt, yp, beta=beta, zero_division=0)
        elif metrica == "precision":
            valores[b] = precision_score(yt, yp, zero_division=0)
        elif metrica == "recall":
            valores[b] = recall_score(yt, yp, zero_division=0)
        else:
            raise ValueError(f"Métrica no bootstrap-able: {metrica}")
    valores = valores[~np.isnan(valores)]
    if valores.size == 0:
        return {"metrica": metrica, "ic_inf": np.nan, "ic_sup": np.nan, "n_boot": 0}
    return {"metrica": metrica,
            "ic_inf": float(np.quantile(valores, alpha / 2)),
            "ic_sup": float(np.quantile(valores, 1 - alpha / 2)),
            "n_boot": int(valores.size), "alpha": alpha, "semilla": semilla}


def mcnemar(y_true, pred_a, pred_b):
    """Contraste de McNemar para dos clasificadores sobre la misma muestra.

    Devuelve el estadístico, el p-valor y la tabla de discordancia. Es el test
    correcto y no un t-test sobre las diferencias de accuracy, porque las
    predicciones de los dos modelos están calculadas sobre los mismos objetos y por
    tanto no son muestras independientes.

    p-valor exacto (binomial) en vez de chi-cuadrado: con las desacuerdas que hay
    aquí (decenas, no miles) la aproximación se aleja.
    """
    from scipy.stats import binomtest

    a = (np.asarray(pred_a) == 1).astype(int)
    b = (np.asarray(pred_b) == 1).astype(int)
    y = np.asarray(y_true)
    aciertos_a = int(((a == y) & (b != y)).sum())   # A acierta, B falla
    aciertos_b = int(((b == y) & (a != y)).sum())   # B acierta, A falla
    n = aciertos_a + aciertos_b
    if n == 0:
        return {"n_disacuerdos": 0, "p": 1.0,
                "nota": "sin discordancias: los dos clasificadores coinciden"}
    p = binomtest(aciertos_a, n, 0.5).pvalue
    return {"a_acierta_b_falla": aciertos_a,
            "b_acierta_a_falla": aciertos_b,
            "n_disacuerdos": n,
            "p": float(p),
            "test": "McNemar exacto (binomial)",
            "nota": "p >= 0.05 no permite afirmar que un modelo sea mejor que otro"}


def linea_base_ap(y_true):
    """PR-AUC de un clasificador que predice la prevalencia de la clase positiva.

    Es el valor que alcanza la average precision de un modelo sin información, y por
    tanto el suelo por debajo del cual cualquier resultado es ruido.
    """
    y = np.asarray(y_true)
    return float(y.mean()) if len(y) else np.nan


def tabla_comparativa(filas, metricas=("F2", "precision", "recall", "PR-AUC", "ROC-AUC")):
    """Pivota una lista de dicts de `resumen_holdout` en una tabla con la etiqueta.

    `etiqueta` es columna y no nota al pie: es la información que decide si dos filas
    son comparables, y enterrarla en una nota es lo que permitió durante meses que
    un F2=0.785 sobre "PHA" conviviera con un F2 sobre otra cosa sin que nadie lo
    notara.
    """
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    cols = [c for c in ["etiqueta", "n", "n_pos", "umbral", *metricas,
                        "TP", "FP", "FN", "TN"] if c in df.columns]
    out = df[cols].copy()
    for c in metricas:
        if c in out.columns:
            out[c] = out[c].round(4)
    return out.sort_values(["etiqueta", "F2"], ascending=[True, False]).reset_index(drop=True)


def media_entre_semillas(registros, columna="F2", agrupar=("etiqueta", "conjunto")):
    """Resume los resultados de las repeticiones: media, sd e IC de la media.

    Este es el número que informa la varianza *entre semillas*: cada fila es un
    reparto de folds distinto y por tanto un experimento independiente. El intervalo
    se calcula sobre las semillas, no sobre las filas de prueba, y por eso no
    duplica el de `intervalo_bootstrap`: son dos fuentes distintas de incertidumbre y
    solo una de ellas suele reportarse.
    """
    df = pd.DataFrame(registros)
    if df.empty or columna not in df.columns:
        return pd.DataFrame()
    g = df.groupby(list(agrupar))[columna]
    out = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
    out = out.rename(columns={"count": "n_repeticiones", "mean": "media",
                              "std": "sd_entre_semillas", "min": "min", "max": "max"})
    out["error_estandar"] = out["sd_entre_semillas"] / np.sqrt(out["n_repeticiones"])
    out["ic95_inf"] = out["media"] - 1.96 * out["error_estandar"]
    out["ic95_sup"] = out["media"] + 1.96 * out["error_estandar"]
    return out
