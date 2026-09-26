"""XAI con SHAP para el pipeline MOID -> PHA.

La pregunta que responde este módulo es qué empuja a un objeto a PHA, no cuánto
acierta el modelo. Importa porque el resultado central del proyecto es una regla de
decisión y, sin explicar, la regla no es auditable por nadie que no haya escrito el
código.

Dos detalles que un `shap.Explainer` a pelo no resuelve:

*   **Se explica la matriz imputada, no la cruda.** El estimador suele ser un
    `Pipeline` (imputador + clasificador). SHAP necesita la matriz que el árbol
    consume, o las importancias se atribuirían a los NaN en lugar de a su valor
    imputado. `preparar_para_shap()` hace esa transformación explícita.
*   **La línea base se declara.** El valor de referencia es la media de la cohorte de
    entrenamiento, no la global: comparar un objeto de 2024 contra la media de
    objetos de 1901 exagera su importancia.
"""

import numpy as np
import pandas as pd

__all__ = [
    "preparar_para_shap",
    "explicar",
    "importancia_global",
    "resumen_objetos",
    "explicar_nuevos",
]

#: Tamaño de la muestra de fondo. Por encima de ~200 filas SHAP es exacto, pero el
#: coste crece con el cuadrado del fondo, así que el sub-muestreo se documenta en la
#: figura en lugar de dejarlo implícito.
FONDO = 200
#: Tope de filas a explicar. Un swarm plot con 37k puntos es ilegible y lento.
MAX_EXPLICADAS = 2000


def _ultimo_estimador(modelo):
    """El estimador con estructura de árbol dentro de un Pipeline, si lo hay."""
    pasos = list(modelo.steps) if hasattr(modelo, "steps") else [("clf", modelo)]
    for _, paso in reversed(pasos):
        if hasattr(paso, "tree_") or hasattr(paso, "get_booster"):
            return paso
    return modelo


def preparar_para_shap(modelo, X, nombres):
    """Aplica el preprocesamiento del Pipeline y devuelve (matriz, estimador).

    Devolver ambas cosas garantiza que se explique exactamente el modelo que se
    evalúa: si se transformara la matriz por un camino distinto al del Pipeline, las
    importancias no podrían corresponder con el modelo entrenado.
    """
    X = np.asarray(X, dtype=float)
    transformador = None
    if hasattr(modelo, "named_steps") and len(modelo.steps) > 1:
        transformador = modelo.steps[0][1]
    if transformador is not None and hasattr(transformador, "transform"):
        X = np.asarray(transformador.transform(X), dtype=float)
    return X, _ultimo_estimador(modelo)


def explicar(modelo, X, nombres, X_fondo, semilla=20, max_amostras=MAX_EXPLICADAS):
    """Valores SHAP y metadatos del cálculo.

    Devuelve un dict porque el coste, el fondo, la semilla y el número de filas
    explicadas son parte del resultado: una tabla de importancias sin ellos no es
    reproducible ni comparable con otra corrida.
    """
    import shap

    X_expl, estimador = preparar_para_shap(modelo, X, nombres)
    X_fondo_expl, _ = preparar_para_shap(modelo, X_fondo, nombres)

    rng = np.random.default_rng(semilla)
    fondo = np.asarray(X_fondo_expl, dtype=float)
    if len(fondo) > FONDO:
        fondo = fondo[rng.choice(len(fondo), FONDO, replace=False)]
    fondo = np.nan_to_num(fondo, copy=False)

    es_arbol = hasattr(estimador, "tree_") or hasattr(estimador, "get_booster")
    if es_arbol:
        explicador = shap.TreeExplainer(estimador, data=fondo,
                                        feature_names=list(nombres))
        metodo, coste = "TreeExplainer", "exacto"
    else:
        explicador = shap.Explainer(estimador.predict_proba, fondo)
        metodo, coste = "Explainer genérico (predict_proba)", "aproximado"

    idx = (rng.choice(len(X_expl), max_amostras, replace=False)
           if len(X_expl) > max_amostras else np.arange(len(X_expl)))
    salida = explicador(X_expl[idx], check_additivity=False)
    # `TreeExplainer` devuelve un `Explanation`, no un ndarray. `np.asarray()` sobre
    # ese objeto no convierte los valores: devuelve un array 0-d del propio
    # `Explanation` y el error sale despues, como un `bad operand type for abs()`
    # en la tabla de importancias, muy lejos de la causa real.
    valores = (salida.values if hasattr(salida, "values")
               else np.asarray(salida))

    # Para un clasificador binario, TreeExplainer puede devolver (n, p) o
    # (n, p, 2). Se toma la clase positiva, que es la que explica la decisión.
    if valores.ndim == 3:
        valores = valores[:, :, -1]

    return {"valores": valores, "X": X_expl[idx], "idx": idx,
            "nombres": list(nombres), "metodo": metodo, "coste": coste,
            "n_fondo": int(len(fondo)), "n_explicado": int(len(idx)),
            "semilla": semilla,
            "nota": ("SHAP se calcula sobre la matriz imputada por el Pipeline: los "
                     "valores de las features sin dato son los imputados, no los "
                     "originales.")}


def importancia_global(resultado):
    """Importancia media absoluta por feature, con su dispersión.

    `mean(|shap|)` es la magnitud del efecto; la dispersión dice si ese efecto lo
    tienen siempre los mismos objetos (una regla) o depende del objeto (interacción).
    """
    sh = np.asarray(resultado["valores"])
    v = np.abs(sh)
    filas = [{"feature": nombre,
              "importancia": float(v[:, i].mean()),
              "p25": float(np.quantile(v[:, i], 0.25)),
              "p75": float(np.quantile(v[:, i], 0.75)),
              "signo_medio": float(np.sign(sh[:, i]).mean())}
             for i, nombre in enumerate(resultado["nombres"])]
    df = pd.DataFrame(filas).sort_values("importancia", ascending=False)
    df.attrs.update(coste=resultado["coste"], n_fondo=resultado["n_fondo"],
                    nota=resultado["nota"])
    return df.reset_index(drop=True)


def resumen_objetos(resultado, y=None, top=3):
    """Las features que más empujan cada objeto a PHA o a no-PHA.

    Es la forma de leer un caso individual: la importancia global no dice por qué
    *este* objeto es un PHA.
    """
    valores = np.asarray(resultado["valores"])
    nombres = resultado["nombres"]
    filas = []
    for k in range(len(valores)):
        c = valores[k]
        for j in np.argsort(-np.abs(c))[:top]:
            filas.append({"fila": k,
                          "y": int(y[k]) if y is not None else None,
                          "feature": nombres[j],
                          "shap": float(c[j]),
                          "valor": float(resultado["X"][k, j]),
                          "empuja_a_pha": bool(c[j] > 0)})
    return pd.DataFrame(filas)


def explicar_nuevos(modelo, X_nuevos, nombres, X_fondo, ids=None, umbral=None,
                    semilla=20):
    """Explicación de candidatos concretos, con la decisión de cada uno.

    Tabla lista para `results/tables/`: una fila por objeto con su probabilidad, si
    supera el umbral, y las dos features que más pesaron. `ids` son las designaciones,
    para que la tabla sea utilizable sin cruzarla con el dataframe a mano.
    """
    r = explicar(modelo, X_nuevos, nombres, X_fondo, semilla=semilla)
    proba = np.asarray(modelo.predict_proba(r["X"])[:, 1])
    valores = np.asarray(r["valores"])
    ids = list(ids) if ids is not None else list(range(len(X_nuevos)))

    filas = []
    for k in range(len(X_nuevos)):
        c = valores[k]
        orden = np.argsort(-np.abs(c))[:2]
        filas.append({
            "objeto": ids[k],
            "probabilidad": float(proba[k]),
            "decision_pha": bool(umbral is not None and proba[k] >= umbral),
            "top1_feature": nombres[orden[0]], "top1_shap": float(c[orden[0]]),
            "top2_feature": nombres[orden[1]], "top2_shap": float(c[orden[1]]),
        })
    df = pd.DataFrame(filas)
    df.attrs.update(coste=r["coste"], n_fondo=r["n_fondo"], nota=r["nota"])
    return df
