"""Regresión de MOID y clasificacion PHA con holdout temporal, semillas y SHAP.

Este es el pipeline principal y el que sustituye a `predict_moid.py` y
`verify_hazard_prediction.py`, que evaluaban sobre datos distintos y cuyos
resultados no podian compararse entre si.

Las cuatro decisiones que definen como hay que leer sus resultados:

1.  **La distancia CAD es calculada, no observada.** JPL la obtiene integrando la
    solucion orbital hasta la fecha del encuentro. Aqui se le llama siempre
    "distancia CAD calculada". No es un matiz de redaccion: un MOID predicho se
    compara con el MOID de catalogo, que es un minimo sobre toda la historia de la
    solucion, mientras que `distnom_min` es el minimo de un subconjunto de
    acercamientos registrados. La regresion se ajusta contra `moid` y el error se
    reporta en la escala del MOID, que es la que decide si un objeto es PHA.
2.  **El holdout es temporal y corta por la fecha del evento.** Un objeto
    descubierto en 2005 con un acercamiento registrado en 2020 no puede estar en la
    prueba de un modelo que dice haber entrenado solo con informacion anterior a
    2015. `agregar_por_objeto(hasta=...)` cierra esa fuga.
3.  **La etiqueta de PHA exige H<=22 ademas de MOID<=0.05.** Con la sola condicion
    geometrica hay 22 237 objetos etiquetados de los que 2 246 son PHAs: 19 991
    quedan marcados como PHA sin serlo. Un modelo entrenado con esa etiqueta
    aprende otra cosa y sus metricas no son comparables con las de la definicion
    oficial.
4.  **Cada numero lleva su incertidumbre.** Se reportan tres fuentes distintas y
    no intercambiables: la dispersión entre semillas, la de los folds de una misma
    repeticion y el bootstrap sobre las filas de prueba. El repositorio solo
    reportaba la segunda, y la llamaba intervalo.

Uso:

    python scripts/pipeline_moid_pha.py
    python scripts/pipeline_moid_pha.py --semillas 5 --sin-ablation
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from neos import (  # noqa: E402
    datos,
    etiquetas,
    evaluacion,
    features as feats,
    modelos,
    xai,
)
from neos.constantes import (  # noqa: E402
    ANIO_CORTE,
    DIR_FIGURAS,
    DIR_TABLAS,
    FEATURES_MOOD,
    FEAT,
    RANDOM_STATE,
    UMBRAL_H,
    UMBRAL_MOID,
)
from neos.semillas import fijar_semillas_todas, generar_semillas  # noqa: E402

datos.configurar_salida_utf8()


# --------------------------------------------------------------------------- #
# Salida
# --------------------------------------------------------------------------- #
def _asegurar_dir(ruta):
    os.makedirs(ruta, exist_ok=True)
    return ruta


def _serializable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(f"No serializable: {type(o)}")


def guardar_json(obj, ruta):
    """Escribe un dict como JSON, convirtiendo lo que numpy no serializa."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_serializable)
    return ruta


# --------------------------------------------------------------------------- #
# Dataset a nivel objeto
# --------------------------------------------------------------------------- #
def construir_objetos(anio_corte=ANIO_CORTE, exigir_pha=True, verificar=True):
    """Dataset a nivel objeto con features calculadas a cada lado del corte.

    Se construyen **dos** tablas y se concatenan, no una sola: la tabla de
    entrenamiento se agrega con los acercamientos anteriores al corte y la de
    prueba con los posteriores. Agregar una sola vez con `hasta=anio_corte` no
    produciría un conjunto de prueba vacío por casualidad: lo produciría siempre,
    porque un objeto al que solo se le han calculado acercamientos previos a 2015 no
    puede haber sido descubierto después de 2015.

    Esa es exactamente la fuga que hace trampo: con una tabla única, las features
    del objeto de prueba incorporarían acercamientos que en 2015 no existían. Con
    dos tablas, cada objeto lleva solo información disponible en su momento.
    """
    print("1. Cargando catalogo y verificando el snapshot...")
    df = datos.cargar_close_approaches(verificar=verificar)
    print(f"   {len(df):,} eventos post-descubrimiento, "
          f"{df.Object.nunique():,} objetos")

    print(f"2. Agregando a nivel objeto en dos tramos (< {anio_corte} y >= {anio_corte})...")
    pre = datos.agregar_por_objeto(df, hasta=anio_corte, exigir_pha=exigir_pha)
    post = datos.agregar_por_objeto(df, desde=anio_corte, exigir_pha=exigir_pha)
    print(f"   {len(pre):,} objetos con acercamientos anteriores a {anio_corte}, "
          f"{len(post):,} posteriores")

    pre["tramo"] = "pre"
    post["tramo"] = "post"
    obj = pd.concat([pre, post], ignore_index=True)

    # Un objeto con acercamientos a ambos lados del corte aparece en las dos
    # tablas. Concatenarlas sin mas deja el mismo objeto dos veces, con features
    # distintas y a veces contradictorias: el modelo lo ve en el folds de
    # entrenamiento con una `distnom_min` y con otra, y aprende a ignorar la que
    # no corresponde. Se queda una sola fila por objeto, la de la ventana que le
    # toca segun cuando se descubrio, que es la unica informacion que existia en
    # su momento.
    n_por_objeto = obj.groupby("Object").size()
    obj = obj.merge(n_por_objeto.rename("n_tramos").reset_index(), on="Object")
    obj = obj[((obj["tramo"] == "pre") & (obj["first_obs_year"] < anio_corte))
              | ((obj["tramo"] == "post") & (obj["first_obs_year"] >= anio_corte))]
    n_duplicados = int((n_por_objeto > 1).sum())
    obj = obj.drop(columns="n_tramos").reset_index(drop=True)
    print(f"   {len(obj):,} filas tras descartar {n_duplicados:,} objetos con datos "
          f"a ambos lados del corte")
    return etiquetas.aplicar_etiquetas(obj)


def comprobar_fugas_temporales(obj, anio_corte=ANIO_CORTE):
    """Verifica las tres condiciones que hacen que el holdout sea temporal.

    Se llama desde `main()` y no desde un test porque las tres condiciones son
    propiedades del dataset construido, no de una funcion: si `agregar_por_objeto`
    cambia, el fallo aparece aqui y no en una metica.

    1.  Un objeto no puede estar en las dos mitades.
    2.  Ninguna fila de la mitad de prueba puede traer acercamientos anteriores al
        corte, y ninguna de entrenamiento posteriores.
    3.  Un objeto no puede aparecer dos veces, con features distintas.
    """
    tr, te = obj["first_obs_year"] < anio_corte, obj["first_obs_year"] >= anio_corte
    if not (tr.any() and te.any()):
        raise AssertionError(
            f"El corte {anio_corte} deja una de las mitades vacia: no hay holdout. "
            f"{int(tr.sum()):,} filas antes, {int(te.sum()):,} despues.")

    ventana_ok = ((obj.loc[tr, "tramo"] == "pre").all()
                  and (obj.loc[te, "tramo"] == "post").all())
    if not ventana_ok:
        malos = obj.loc[tr & (obj["tramo"] != "pre"), "Object"].tolist()[:5]
        raise AssertionError(
            f"Hay filas cuyo tramo no corresponde a su cohorte, por ejemplo {malos}. "
            "Eso significa que las features incluyen acercamientos de la otra ventana.")

    duplicados = obj["Object"][obj["Object"].duplicated()].unique()
    if len(duplicados):
        raise AssertionError(
            f"{len(duplicados):,} objetos aparecen mas de una vez, por ejemplo "
            f"{list(duplicados[:5])}. El mismo objeto con dos juegos de features "
            "hace que el modelo memorice en vez de generalizar.")
    return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_objetos": int(obj["Object"].nunique())}


def indices_temporales(obj, m, anio_corte=ANIO_CORTE):
    """Índices de entrenamiento y prueba por año de descubrimiento, sin mezcla.

    Se reparte por `first_obs_year` y no por la fecha del evento porque lo que
    define la disponibilidad de información de un objeto es cuándo se descubrió.
    Un objeto descubierto en 2016 cuyo acercamiento más cercano es de 2013 es un
    objeto al que en 2015 no se le podía haber calculado nada.
    """
    anios = obj.loc[m.idx, "first_obs_year"].to_numpy()
    return (np.where(anios < anio_corte)[0],
            np.where(anios >= anio_corte)[0])


# --------------------------------------------------------------------------- #
# Regresion de MOID
# --------------------------------------------------------------------------- #
def _regresor_moid(semilla, modelo="XGBoost"):
    """Regresor de MOID con el imputador dentro del pipeline, no antes."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if modelo == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingRegressor

        reg = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, random_state=semilla)
    else:
        from xgboost import XGBRegressor

        reg = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                           subsample=0.9, random_state=semilla, n_jobs=1)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("reg", reg)])


def evaluar_regresion_moid(obj, features=None, anio_corte=ANIO_CORTE,
                           semilla=RANDOM_STATE, verbose=True):
    """Metricas de la regresion de MOID con holdout temporal.

    El objetivo es `moid` (minimo sobre toda la historia orbital) y el error se
    reporta en au. Un regresor que predijera el minimo del subconjunto de
    acercamientos y se evaluara contra el tendria un error mucho menor, y ese error
    no diria nada sobre si el objeto va a pasar cerca: va a pasar cerca si en algun
    momento de los proximos 100 anos su MOID baja de 0.05, no si su
    acercamiento ya registrado fue el mas cercano de los cinco que ocurrieron.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    features = list(FEATURES_MOOD) if features is None else list(features)
    m = feats.construir_X(obj, features, etiqueta="et_pha")
    idx_tr, idx_te = indices_temporales(obj, m, anio_corte)

    pipeline = _regresor_moid(semilla)
    pipeline.fit(m.X[idx_tr], obj.loc[m.idx[idx_tr], "moid"].to_numpy())
    y_verdad = obj.loc[m.idx[idx_te], "moid"].to_numpy()
    pred = pipeline.predict(m.X[idx_te])

    metricas = {
        "etiqueta": "MOID",
        "corte": "temporal",
        "n_entren": int(len(idx_tr)),
        "n_prueba": int(len(idx_te)),
        "mae_au": float(mean_absolute_error(y_verdad, pred)),
        "r2": float(r2_score(y_verdad, pred)),
        "rmse_au": float(np.sqrt(np.mean((y_verdad - pred) ** 2))),
        "semilla": semilla,
        "features": " + ".join(features),
    }
    if verbose:
        print(f"   MAE={metricas['mae_au']:.4f} au  R2={metricas['r2']:.3f}  "
              f"n_prueba={metricas['n_prueba']:,}")
    return {"metricas": metricas, "pipeline": pipeline, "idx_prueba": idx_te,
            "y_verdad": y_verdad, "pred": pred}


# --------------------------------------------------------------------------- #
# Clasificacion PHA
# --------------------------------------------------------------------------- #
def _tres_folds(cv, m, y, semilla):
    """Reparte cada fold de CV en `i_in` / `i_val` / `i_te`.

    El umbral se elige en `i_val` y el modelo se evalua en `i_te`, asi que los tres
    conjuntos tienen que ser disjuntos. La version anterior ajustaba el modelo con
    `i_in + i_val` y despues elegia el umbral mirando `i_val`: las metricas de
    `i_te` seguian sin estar contaminadas, pero el umbral se calibraba sobre
    predicciones in-sample, que es un sesgo optimista distinto del que se queria
    evitar y mas dificil de detectar.
    """
    from sklearn.model_selection import train_test_split

    for fold, (i_tr, i_te) in enumerate(cv.split(m.X, y)):
        i_in, i_val = train_test_split(i_tr, test_size=0.25,
                                       random_state=semilla + fold,
                                       stratify=y[i_tr])
        yield fold, np.asarray(i_in), np.asarray(i_val), np.asarray(i_te)


def evaluar_clasificacion_pha(obj, features=None, anio_corte=ANIO_CORTE,
                              etiqueta="et_pha", n_semillas=10, verbose=True):
    """Clasificador de PHA con CV repetido y umbral F2 elegido en fold aparte.

    El umbral se elige en un fold de validacion interno y se aplica al fold de
    prueba. Ajustarlo sobre la prueba convierte la prueba en entrenamiento y la
    metrica deja de ser una estimacion de nada.

    Se repite sobre semillas distintas en vez de usar `n_repeats` alto en un solo
    CV: las repeticiones internas comparten datos de entrenamiento, asi que su
    dispersion mezcla dos fuentes. Ademas se hace sobre la poblacion completa con
    CV aleatorio, no con el corte temporal, porque la pregunta "¿sabe el modelo
    distinguir PHA de no-PHA?" y la pregunta "¿sabe decidir sobre NEOs que acaba
    de aparecer?" no tienen la misma respuesta, y confundirlas fue como se
    reporto un 0.999 de accuracy que no significaba nada.
    """
    features = list(FEATURES_MOOD) if features is None else list(features)
    m = feats.construir_X(obj, features, etiqueta=etiqueta)
    feats.comprobar_clase(m.X, m.y, razon_min=0.002, minimo=200)
    y = m.y
    clave = m.etiqueta
    meta = etiquetas.ETIQUETAS[clave]

    por_semilla, texturas = [], []
    for semilla in generar_semillas(n_semillas):
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
        y_test, probas, f2s, aps, recalls, precisions, rocs, umbrales = (
            [], [], [], [], [], [], [], [])
        for fold, i_in, i_val, i_te in _tres_folds(cv, m, y, semilla):
            mod = modelos.crear_modelos(y[i_in], random_state=semilla + fold)["XGBoost"]
            mod.fit(m.X[i_in], y[i_in])
            umbral = modelos.umbral_por_f2(
                y[i_val], mod.predict_proba(m.X[i_val])[:, 1])["umbral"]
            proba = mod.predict_proba(m.X[i_te])[:, 1]
            r = evaluacion_resumida(y[i_te], proba, umbral)
            y_test.append(y[i_te]); probas.append(proba)
            f2s.append(r["F2"]); aps.append(r["PR-AUC"])
            recalls.append(r["recall"]); precisions.append(r["precision"])
            rocs.append(roc_auc_score(y[i_te], proba))
            umbrales.append(umbral)

        y_test = np.concatenate(y_test); probas = np.concatenate(probas)
        por_semilla.append({
            "etiqueta": clave, "corte": "aleatorio 5-fold",
            "definicion": meta["definicion"], "semilla": semilla,
            "n": int(len(y_test)), "n_pos": int(y_test.sum()),
            "prevalencia": float(y_test.mean()),
            "F2": float(np.mean(f2s)), "PR-AUC": float(np.mean(aps)),
            "recall": float(np.mean(recalls)), "precision": float(np.mean(precisions)),
            "ROC-AUC": float(np.mean(rocs)),
            "umbral_mediano": float(np.median(umbrales)),
        })
        texturas.append((y_test, probas))

    if verbose:
        fs = [r["F2"] for r in por_semilla]
        aps = [r["PR-AUC"] for r in por_semilla]
        prev = por_semilla[0]["prevalencia"]
        print(f"   F2={np.mean(fs):.3f} ± {np.std(fs):.3f}  "
              f"PR-AUC={np.mean(aps):.3f} ± {np.std(aps):.3f}  "
              f"(linea base AP = {100 * prev:.2f}%)")
    return {"por_semilla": por_semilla, "texturas": texturas, "objetos": m,
            "etiqueta": clave}


def evaluacion_resumida(y_true, y_score, umbral, beta=modelos.BETA):
    """Métricas de un fold. Envuelve `neos.evaluacion` para no recalcular el train."""
    from neos.evaluacion import _metricas

    return _metricas(y_true, y_score, umbral, beta=beta)


# --------------------------------------------------------------------------- #
# Ablacion
# --------------------------------------------------------------------------- #
def ablar_features(obj, conjuntos=None, etiqueta="et_pha", n_semillas=3,
                   verbose=True):
    """Metrica de PHA por conjunto de features, con el resto del protocolo fijo.

    La ablacion responde a la pregunta que el repositorio no podia responder: si
    `v_inf` aporta señal mas alla de la distancia ya calculada. Sin ella, un
    resultado alto con el conjunto completo es indistinguible de haber aprendido la
    regla de un umbral sobre la distancia. `sin-dist` frente a `proxy-1feat` es la
    comparacion decisiva: si no los supera, no hay senal cinematica independiente.
    """
    conjuntos = list(FEAT) if conjuntos is None else list(conjuntos)
    filas = []
    for nombre in conjuntos:
        cols = FEAT[nombre] if isinstance(nombre, str) else list(nombre)
        for semilla in generar_semillas(n_semillas):
            m = feats.construir_X(obj, cols, etiqueta=etiqueta)
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
            f2s, aps, recalls = [], [], []
            for _, i_in, i_val, i_te in _tres_folds(cv, m, m.y, semilla):
                mod = modelos.crear_modelos(m.y[i_in], random_state=semilla)["XGBoost"]
                mod.fit(m.X[i_in], m.y[i_in])
                # El umbral se elige en `i_val`. La version anterior lo elegia con
                # `i_te`, que ademas es exactamente el conjunto cuya F2 se reportaba:
                # cada numero de la tabla estaba optimizado sobre si mismo, y por eso
                # la ablacion daba siempre un F2 mayor cuanto menos features tinha.
                umbral = modelos.umbral_por_f2(
                    m.y[i_val], mod.predict_proba(m.X[i_val])[:, 1])["umbral"]
                r = evaluacion_resumida(m.y[i_te],
                                         mod.predict_proba(m.X[i_te])[:, 1], umbral)
                f2s.append(r["F2"]); aps.append(r["PR-AUC"]); recalls.append(r["recall"])
            filas.append({"conjunto": nombre, "n_features": len(cols),
                          "features": " + ".join(cols), "semilla": semilla,
                          "F2": float(np.mean(f2s)), "PR-AUC": float(np.mean(aps)),
                          "recall": float(np.mean(recalls))})
        if verbose:
            sub = [f for f in filas if f["conjunto"] == nombre]
            print(f"   {nombre:14s} p={sub[0]['n_features']}  "
                  f"F2={np.mean([f['F2'] for f in sub]):.3f}  "
                  f"PR-AUC={np.mean([f['PR-AUC'] for f in sub]):.3f}")
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# XAI
# --------------------------------------------------------------------------- #
def explicar_con_shap(obj, features=None, anio_corte=ANIO_CORTE,
                      semilla=RANDOM_STATE, max_explicadas=2000):
    """SHAP del regresor de MOID y del clasificador de PHA, en ese orden.

    El regresor va primero porque es el eslabon que decide el umbral: entender que
    mueve el MOID predicho es previo a entender por que un objeto se clasifica.
    Se explica sobre la matriz imputada por el pipeline, que es la que consume el
    arbol: explicar la matriz cruda atribuiria a los NaN lo que en realidad es el
    valor imputado.
    """
    features = list(FEATURES_MOOD) if features is None else list(features)
    m = feats.construir_X(obj, features, etiqueta="et_pha")
    idx_tr, idx_te = indices_temporales(obj, m, anio_corte)
    out = {}

    reg = _regresor_moid(semilla)
    reg.fit(m.X[idx_tr], obj.loc[m.idx[idx_tr], "moid"].to_numpy())
    X_te, arbol = xai.preparar_para_shap(reg, m.X[idx_te], features)
    r = xai.explicar(arbol, X_te, features, m.X[idx_tr], semilla=semilla,
                     max_amostras=max_explicadas)
    out["moid"] = {"importancia": xai.importancia_global(r),
                   "resumen": xai.resumen_objetos(r), "metodo": r["metodo"],
                   "coste": r["coste"], "n_explicado": r["n_explicado"]}

    clas = modelos.crear_modelos(m.y, random_state=semilla)["XGBoost"]
    clas.fit(m.X[idx_tr], m.y[idx_tr])
    X_te_c, arbol_c = xai.preparar_para_shap(clas, m.X[idx_te], features)
    rc = xai.explicar(arbol_c, X_te_c, features, m.X[idx_tr], semilla=semilla,
                      max_amostras=max_explicadas)
    out["pha"] = {"importancia": xai.importancia_global(rc),
                  "resumen": xai.resumen_objetos(rc), "metodo": rc["metodo"],
                  "coste": rc["coste"], "n_explicado": rc["n_explicado"]}
    return out


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figuras(obj, eval_moid, eval_pha, ablaciones=None, shap_res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    _asegurar_dir(DIR_FIGURAS)
    rutas = {}

    y, p = eval_moid["y_verdad"], eval_moid["pred"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, p, s=8, alpha=0.25, c="#4C72B0", edgecolors="none")
    lim = max(y.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="y = x")
    ax.axhline(UMBRAL_MOID, color="darkorange", ls=":", lw=1.5,
               label=f"umbral PHA = {UMBRAL_MOID} au")
    ax.axvline(UMBRAL_MOID, color="darkorange", ls=":", lw=1.5)
    ax.set_xlabel("MOID de catalogo (au)")
    ax.set_ylabel("MOID predicho (au)")
    ax.set_title("Regresion de MOID con holdout temporal\n"
                 "(features = distancia CAD calculada, v_inf, H, incertidumbre)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    rutas["moid_regresion"] = os.path.join(DIR_FIGURAS, "moid_regresion_temporal.png")
    fig.savefig(rutas["moid_regresion"], dpi=150)
    plt.close(fig)

    z = pd.cut(y, bins=[0, UMBRAL_MOID, 0.10, 0.20, 0.50, np.inf],
               labels=["<0.05 (PHA)", "0.05-0.10", "0.10-0.20", "0.20-0.50", ">0.50"])
    err = np.abs(y - p)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=z, y=err, ax=ax, color="#A6C8E0")
    ax.axhline(UMBRAL_MOID / 2, color="gray", ls="--", lw=1,
               label=f"{UMBRAL_MOID / 2} au (margen de la etiqueta)")
    ax.set_xlabel("zona de MOID de catalogo (au)")
    ax.set_ylabel("|error| (au)")
    ax.set_title("Error de la regresion de MOID por zona orbital")
    ax.legend(fontsize=9)
    fig.tight_layout()
    rutas["moid_error_zona"] = os.path.join(DIR_FIGURAS, "moid_error_por_zona.png")
    fig.savefig(rutas["moid_error_zona"], dpi=150)
    plt.close(fig)

    y_test, probas = eval_pha["texturas"][0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for valor, color, nombre in ((0, "#8C8C8C", "no PHA"), (1, "#C44E52", "PHA")):
        sel = y_test == valor
        if sel.any():
            sns.histplot(probas[sel], bins=40, ax=ax, color=color, alpha=0.6,
                         label=f"{nombre} (n={int(sel.sum()):,})", stat="density")
    ax.axvline(0.5, color="black", ls="--", lw=1.2, label="umbral 0.5")
    ax.set_xlabel("probabilidad de PHA")
    ax.set_ylabel("densidad")
    ax.set_title("Probabilidad de PHA por clase real (CV 5-fold sobre toda la poblacion)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    rutas["pha_prob"] = os.path.join(DIR_FIGURAS, "pha_probabilidad_cv.png")
    fig.savefig(rutas["pha_prob"], dpi=150)
    plt.close(fig)

    if ablaciones is not None and not ablaciones.empty:
        agg = ablaciones.groupby("conjunto", as_index=False)[["F2", "PR-AUC"]].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg["F2"], 0.4, label="F2")
        ax.bar(x + 0.2, agg["PR-AUC"], 0.4, label="PR-AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["conjunto"], rotation=25, ha="right")
        ax.set_ylabel("metrica")
        ax.set_title("Ablacion de features (XGBoost, CV 5-fold, media de 3 semillas)")
        ax.legend()
        fig.tight_layout()
        rutas["ablation"] = os.path.join(DIR_FIGURAS, "ablation_features.png")
        fig.savefig(rutas["ablation"], dpi=150)
        plt.close(fig)

    for clave, sufijo, titulo in (
            ("moid", "shap_moid", "Regresor de MOID"),
            ("pha", "shap_pha", "Clasificador de PHA")):
        if not shap_res or clave not in shap_res:
            continue
        imp = shap_res[clave]["importancia"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(imp["feature"], imp["importancia"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_xlabel("E[|SHAP|]")
        ax.set_title(f"Importancia SHAP — {titulo} "
                     f"({shap_res[clave]['coste']}, "
                     f"n={shap_res[clave]['n_explicado']:,})")
        fig.tight_layout()
        rutas[sufijo] = os.path.join(DIR_FIGURAS, f"{sufijo}_importance.png")
        fig.savefig(rutas[sufijo], dpi=150)
        plt.close(fig)

    return rutas


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--anio-corte", type=int, default=ANIO_CORTE)
    p.add_argument("--semillas", type=int, default=10)
    # `choices` sobre las columnas de las etiquetas, no sobre el dict entero: con el
    # dict, `--help` imprimia cuatro objetos de tres lineas en lugar de cuatro
    # nombres, y ningun error de tecleo daba un mensaje util.
    p.add_argument("--etiqueta", default="et_pha",
                   choices=sorted(m["columna"] for m in etiquetas.ETIQUETAS.values()),
                   help="columna de etiqueta a predecir (por defecto: et_pha, la "
                        "definicion oficial reconstruida)")
    p.add_argument("--sin-ablation", action="store_true")
    p.add_argument("--sin-shap", action="store_true")
    p.add_argument("--sin-verificar", action="store_true")
    args = p.parse_args(argv)

    sem = fijar_semillas_todas(RANDOM_STATE)
    print(f"Semillas globales: {sem}")
    print(f"Corte temporal: descubrimiento < {args.anio_corte} entrena, "
          f">= {args.anio_corte} prueba\n")
    _asegurar_dir(DIR_TABLAS)
    _asegurar_dir(DIR_FIGURAS)

    t0 = time.time()
    obj = construir_objetos(anio_corte=args.anio_corte, verificar=not args.sin_verificar)
    fuga = comprobar_fugas_temporales(obj, anio_corte=args.anio_corte)
    print(f"   Holdout verificado: {fuga['n_train']:,} filas de entrenamiento, "
          f"{fuga['n_test']:,} de prueba, {fuga['n_objetos']:,} objetos sin repetir")
    print(f"   (dataset en {time.time() - t0:.1f} s)\n")

    print("4. Regresion de MOID (holdout temporal)...")
    ev_moid = evaluar_regresion_moid(obj, anio_corte=args.anio_corte)
    print()

    print("5. Clasificacion PHA (CV 5-fold, umbral F2 por fold)...")
    ev_pha = evaluar_clasificacion_pha(obj, anio_corte=args.anio_corte,
                                       etiqueta=args.etiqueta,
                                       n_semillas=args.semillas)
    print()

    abl = None
    if not args.sin_ablation:
        print("6. Ablacion de features (3 semillas)...")
        abl = ablar_features(obj, etiqueta=args.etiqueta)
        print()

    shap_res = {}
    if not args.sin_shap:
        print("7. SHAP (regresor de MOID y clasificador de PHA)...")
        shap_res = explicar_con_shap(obj, anio_corte=args.anio_corte)
        for clave in shap_res:
            top = shap_res[clave]["importancia"].iloc[0]
            print(f"   {clave}: la feature que mas pesa es {top['feature']!r} "
                  f"(E[|SHAP|]={top['importancia']:.3f})")
        print()

    print("8. Figuras...")
    for nombre, ruta in figuras(obj, ev_moid, ev_pha, abl, shap_res).items():
        print(f"   {nombre}: {os.path.basename(ruta)}")

    print("\n9. Tablas...")
    # Una tabla por bloque, no una comun. Apilarlas en un solo CSV produce filas
    # donde la mitad de las columnas va vacia y el resto tiene otro significado:
    # la fila de MOID no tiene F2, la de ablacion no tiene etiqueta, y las metricas
    # de una regresion quedan en la misma columna que las de una clasificacion.
    # Es la razon por la que un F2 de PHA y otro de `MOID<=0.05` convivieron sin
    # que se notara que no son comparables.
    rutas = {}
    rutas["regresion_moid"] = _guardar(
        pd.DataFrame([ev_moid["metricas"]]), "regresion_moid.csv")
    rutas["clasificacion_pha"] = _guardar(
        evaluacion.tabla_comparativa(ev_pha["por_semilla"],
                                    metricas=("F2", "PR-AUC", "ROC-AUC",
                                              "precision", "recall")),
        "clasificacion_pha.csv")
    if abl is not None:
        resumen = evaluacion.media_entre_semillas(abl.to_dict("records"),
                                                  agrupar=("conjunto", "n_features"))
        rutas["ablation_features"] = _guardar(
            resumen, "ablation_features.csv")
        rutas["ablation_detalle"] = _guardar(abl, "ablation_detalle.csv")

    for clave in (shap_res or {}):
        rutas[f"shap_{clave}_importance"] = _guardar(
            shap_res[clave]["importancia"], f"shap_{clave}_importance.csv")
        rutas[f"shap_{clave}_objetos"] = _guardar(
            shap_res[clave]["resumen"], f"shap_{clave}_objetos.csv")

    # La uncertainty que no cabe en una tabla de metricas por punto: el bootstrap
    # sobre las filas de prueba y el contraste entre modelos.
    y_cv, probas_cv = ev_pha["texturas"][0]
    umbral_cv = float(np.median([r["umbral_mediano"] for r in ev_pha["por_semilla"]]))
    intervalo = evaluacion.intervalo_bootstrap(y_cv, probas_cv, umbral_cv,
                                               metrica="F2", semilla=RANDOM_STATE)
    print(f"   F2 con bootstrap 95 %: [{intervalo['ic_inf']:.3f}, "
          f"{intervalo['ic_sup']:.3f}] sobre {intervalo['n_boot']:,} remuestreos")
    # El bootstrap es condicional a la muestra de prueba; la dispersion entre
    # semillas es la otra fuente. Se reportan las dos y no se suman.
    entre_semillas = evaluacion.media_entre_semillas(
        ev_pha["por_semilla"], agrupar=("etiqueta",))
    if not entre_semillas.empty:
        print(f"   F2 entre semillas: {entre_semillas.loc[0, 'media']:.3f} "
              f"± {entre_semillas.loc[0, 'sd_entre_semillas']:.3f} "
              f"({int(entre_semillas.loc[0, 'n_repeticiones'])} repeticiones)")

    guardar_json({
        "anio_corte": args.anio_corte,
        "etiqueta": args.etiqueta,
        "semillas": list(generar_semillas(args.semillas)),
        "semillas_derivadas": sem,
        "features_moid": list(FEATURES_MOOD),
        "conjuntos_ablation": list(FEAT),
        "n_objetos": int(len(obj)),
        "umbrales": {"moid": UMBRAL_MOID, "h": UMBRAL_H},
        "bootstrap_f2_95": intervalo,
        "generado_utc": datetime.now(timezone.utc).isoformat(),
    }, os.path.join(DIR_TABLAS, "pipeline_config.json"))
    for nombre in rutas.values():
        print(f"   {os.path.basename(nombre)}")
    print(f"   pipeline_config.json\nTotal: {time.time() - t0:.1f} s")


def _guardar(df, nombre):
    """Escribe una tabla en `results/tables/` y devuelve su ruta."""
    ruta = os.path.join(DIR_TABLAS, nombre)
    df.to_csv(ruta, index=False)
    return ruta


if __name__ == "__main__":
    main()
