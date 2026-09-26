"""Construcción de matrices de features y detección de fuga temporal.

Dos problemas que estaban repartidos por los scripts:

*   **Imputación fuera del CV.** `SimpleImputer(strategy='median')` se ajustaba una
    vez sobre todo el dataframe, así que la mediana de cada feature includa los
    valores del conjunto de prueba. Con features correlacionadas y poblaciones
    desiguales eso no es neutro: infla la métrica. Aquí el imputador se construye
    *dentro* de cada fold, como el resto del pipeline.
*   **Fuga temporal.** La comprobación que faltaba no era "qué columnas uso", sino
    "usé información del futuro". `auditar_fugas()` calcula, para cada feature, su
    correlación con la fecha de descubrimiento, y en particular la coriación de
    `n_appro` con la cohorte, que es donde el holdout temporal se rompía.
"""

import numpy as np
import pandas as pd

from neos.constantes import ETIQUETAS_FEATURES, FEAT, FEATURES_MOOD

__all__ = [
    "construir_X",
    "columnas_de",
    "columna_de",
    "columna_etiqueta",
    "nombres_amigables",
    "auditar_fugas",
    "comprobar_clase",
]

#: Features que dependen del número de eventos registrados y por tanto de la
#: exposición del sondeo. No son física del objeto.
EXPOSICION = {"n_appro"}


def columna_de(nombre, obj):
    """Resuelve el nombre interno de una feature a su columna real.

    Las features se nombran en el código (`distnom_min`) pero viven en el dataframe
    con el nombre de la API (`CA DistanceNominal (au)`). Cada vez que se cruzaba
    una de las dos capas, el error era un `KeyError` de tres páginas más allá.
    """
    if nombre in obj.columns:
        return nombre
    alias = {
        "distnom_min": "CA DistanceNominal (au)",
        "distmin_min": "CA DistanceMinimum (au)",
        "vinf_closest": "V infinity(km/s)",
        "vinf_max": "V infinity(km/s)",
        "H_obs": "H(mag)",
        "H_sbdb": "H_SBDB(mag)",
        "dist_unc_med": "dist_unc",
    }
    destino = alias.get(nombre)
    if destino and destino in obj.columns:
        return destino
    raise KeyError(
        f"La feature {nombre!r} no está en el dataframe. Disponibles: "
        f"{sorted(c for c in obj.columns if c in set(alias.values()) | set(alias))}")


def columnas_de(nombre_ensemble, obj):
    """Las columnas reales de un conjunto de features, validado contra el dataframe."""
    if not isinstance(nombre_ensemble, (list, tuple)):
        raise TypeError(
            f"{nombre_ensemble!r} no es un conjunto de features: se esperaba una "
            "lista como FEATURES_MOOD o una clave de FEAT.")
    faltan = [f for f in nombre_ensemble if f not in ETIQUETAS_FEATURES]
    if faltan:
        raise KeyError(
            f"Features no declaradas en ETIQUETAS_FEATURES: {faltan}. "
            "Añádela ahí antes de usarla, o el eje de la figura saldrá sin nombre.")
    return [columna_de(f, obj) for f in nombre_ensemble]


def nombres_amigables(feature):
    """Texto para ejes, títulos y tablas de importancia."""
    if isinstance(feature, (list, tuple)):
        return " + ".join(ETIQUETAS_FEATURES.get(f, f) for f in feature)
    return ETIQUETAS_FEATURES.get(feature, feature)


def columna_etiqueta(etiqueta):
    """Resuelve una etiqueta a su columna, aceptando clave o nombre de columna.

    `etiqueta` puede ser la clave de `ETIQUETAS` ("PHA") o el nombre de la columna
    ("et_pha"). Las dos formas existen ya en el código: los notebooks usan la clave y
    los scripts la columna, y la que no se aceptaba era un `KeyError` con un mensaje
    que no decía qué etiquetas había.
    """
    from neos.etiquetas import ETIQUETAS

    if etiqueta in ETIQUETAS:
        return ETIQUETAS[etiqueta]["columna"], etiqueta
    for clave, meta in ETIQUETAS.items():
        if meta["columna"] == etiqueta:
            return meta["columna"], clave
    disponibles = sorted(
        list(ETIQUETAS) + [m["columna"] for m in ETIQUETAS.values()])
    raise KeyError(f"Etiqueta desconocida: {etiqueta!r}. Disponibles: {disponibles}")


def construir_X(obj, features, con_etiqueta=True, etiqueta="PHA", dropna=False):
    """Matriz de diseño X e y para un conjunto de features.

    Devuelve un objeto con `.X`, `.y`, `.feature_names` y `.idx`: devolver una
    tupla `(X, y)` obligaba a recomputar la lista de columnas en el llamante, que
    es donde aparecían los `X[:, 3]` sin nombre.

    `dropna=False` por defecto a propósito. Los modelos llevan un `SimpleImputer`
    dentro del pipeline precisamente para tratar las features sin dato; con
    `dropna=True` las filas que falta algo se eliminaban aqui y el imputador no
    tenia nada que hacer, asi que la imputacion no se podia ni medir ni auditar.
    Solo se descartan las filas cuya *etiqueta* es NaN, porque esas no tienen
    destino: sin `y` no hay nada que predecir.
    """
    columnas = columnas_de(features, obj)
    col_et, clave_et = (columna_etiqueta(etiqueta) if con_etiqueta else (None, None))
    sub = obj.loc[:, list(columnas) + ([col_et] if con_etiqueta else [])].copy()
    n_antes = len(sub)
    if con_etiqueta:
        sub = sub[sub[col_et].notna()]
    if dropna:
        sub = sub.dropna()
    y = sub[col_et].astype(int).to_numpy() if con_etiqueta else None
    m = MatrizDesign(sub[columnas].to_numpy(dtype=float), y, list(features),
                     sub.index, n_descartados=len(obj) - n_antes)
    m.etiqueta = clave_et
    m.n_filas_sin_y = n_antes - len(sub)
    return m


class MatrizDesign:
    """X, y, nombres de feature e índice original, en un solo objeto.

    Los scripts anteriores pasaban `X` e `y` sueltos y luego reconstruían los
    nombres desde una constante global, lo que rompía en cuanto un script usaba un
    subconjunto de features distinto del de otro.
    """

    def __init__(self, X, y, feature_names, idx, n_descartados=0, etiqueta=None):
        self.X = X
        # `y` es un ndarray y `idx` un Index de pandas. Mezclarlos hacia que
        # `y[i_tr]` con indices de numpy lanzara un KeyError listando los indices:
        # pandas interpreta una lista como acceso .loc por etiqueta, y los numeros de
        # fila de X no son etiquetas. Que la Serie devolviera en silencio filas
        # equivocadas seria peor que este error, asi que la invariante es que `y`
        # sea siempre ndarray.
        self.y = np.asarray(y)
        self.feature_names = list(feature_names)
        self.idx = idx
        self.n_descartados = n_descartados
        self.n_filas_sin_y = 0
        self.etiqueta = etiqueta

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return (f"MatrizDesign(n={len(self)}, p={self.X.shape[1]}, "
                f"features={self.feature_names})")

    def con_etiqueta(self, etiqueta, obj):
        """Reetiqueta sin rehacer la imputación (la etiqueta no es una feature)."""
        col_et, clave_et = columna_etiqueta(etiqueta)
        self.y = obj.loc[self.idx, col_et].astype(int).to_numpy()
        self.etiqueta = clave_et
        return self

    def train_test_idx(self, test_size=0.2, semilla=20, temporal=False, anio_col=None):
        """Índices de entrenamiento y prueba, con o sin respecto al tiempo.

        Con `temporal=True` no hay mezcla al azar: se entrena con lo anterior al
        corte y se evalúa con lo posterior. `anio_col` es la columna de cohorte
        (por defecto el año del evento, no el del descubrimiento).
        """
        from sklearn.model_selection import train_test_split

        if temporal:
            if anio_col is None:
                raise ValueError("temporal=True necesita anio_col")
            anios = pd.Series(self.idx.map(anio_col) if callable(anio_col) else anio_col)
            corte = anios.quantile(1 - test_size)
            mask_prueba = anios >= corte
            return np.where(~mask_prueba.to_numpy())[0], np.where(mask_prueba.to_numpy())[0]
        idx = np.arange(len(self))
        return train_test_split(idx, test_size=test_size, random_state=semilla,
                                stratify=self.y)


def auditar_fugas(obj, features=None, col_anio="first_obs_year", etiqueta="PHA"):
    """Correlación de cada feature con la cohorte: así se detecta la fuga.

    Una correlación alta con la cohorte no es en sí un error (la población cambia),
    pero sí lo es si la feature entra en el modelo y el holdout es temporal: el
    modelo aprende el cambio del muestreo en vez del objeto. Este informe es el que
    justifica dejar `n_appro` fuera del pipeline principal.
    """
    # `FEAT` esta indexado por nombre de conjunto, no por feature. `list(FEAT)` da
    # ["kin+size", "kin-only", ...], y `columna_de("kin+size", obj)` falla al primer
    # elemento: el informe que justifica excluir `n_appro` no se podia ejecutar.
    if features is None:
        features = sorted({f for cols in FEAT.values() for f in cols}
                          | set(FEATURES_MOOD))
    else:
        features = list(features)
    col_et, _ = columna_etiqueta(etiqueta)
    filas = []
    for nombre in features:
        col = columna_de(nombre, obj)
        serie = pd.to_numeric(obj[col], errors="coerce")
        con = pd.DataFrame({"f": serie, "a": obj[col_anio], "y": obj[col_et]})
        con = con.dropna()
        filas.append({
            "feature": nombre,
            "nombre": nombres_amigables(nombre),
            "exp_posicion": nombre in EXPOSICION,
            "n": len(con),
            "corr_anio": con["f"].corr(con["a"], method="spearman"),
            "corr_etiqueta": con["f"].corr(con["y"], method="spearman"),
            "mediana": con["f"].median(),
        })
    informe = pd.DataFrame(filas).sort_values("corr_anio", key=lambda s: s.abs(),
                                               ascending=False)
    informe = informe.reset_index(drop=True)
    informe.attrs["nota"] = (
        "corr_anio alto + exp_posicion=True indica que la feature codifica la "
        "exposición del sondeo, no el objeto: usarla con holdout temporal hace que "
        "el test mida un cambio de muestreo.")
    return informe


def comprobar_clase(X, y, razon_min=0.05, minimo=50):
    """Verifica que haya filas y clases suficientes para que la métrica signifique algo.

    Un holdout temporal con 50 filas de prueba y 3 positivos no produce una métrica
    interpretable, y el error no aparece como excepción: aparece como un
    `precision=0.00` silencioso que luego se copia a una tabla.
    """
    X, y = np.asarray(X), np.asarray(y)
    n = len(y)
    if n == 0:
        raise ValueError("No hay filas: la matriz de diseño está vacía.")
    if n < minimo:
        raise ValueError(f"Solo {n} filas; el mínimo para un holdout es {minimo}.")
    proporcion = float(y.mean())
    if not (razon_min <= proporcion <= 1 - razon_min):
        raise ValueError(
            f"Proporción de clase minoritaria {proporcion:.4f} fuera de "
            f"[{razon_min}, {1 - razon_min}]: la métrica será inestable.")
    return {"n": n, "p_features": X.shape[1] if X.ndim == 2 else 0,
            "proporcion_positiva": proporcion}
