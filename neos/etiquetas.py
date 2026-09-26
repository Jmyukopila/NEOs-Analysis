"""Definiciones de las etiquetas y de la nomenclatura que las rodea.

Este módulo existe por una confusión concreta que ya se había propagado a todo
el repositorio: durante un tiempo se llamó "PHA" a la condición `MOID <= 0.05 au`,
que es **media** definición de PHA. La definición oficial exige además
`H <= 22 mag`, y en el catálogo del proyecto esa condición adicional baja la
prevalencia del 59 % al 6 %.

Consecuencia práctica: 22 233 objetos de 37 405 satisfacen `MOID <= 0.05`, pero
solo 2 246 son PHAs. Nombrarlos igual convierte una diferencia de magnitud en un
matiz de redacción.

Las funciones de aquí devuelven siempre las tres etiquetas por separado y con
nombre propio, de modo que cada tabla pueda declarar cuál está usando.
"""

import pandas as pd

from neos.constantes import UMBRAL_H, UMBRAL_MOID

__all__ = [
    "ETIQUETAS",
    "nombre_etiqueta",
    "etiqueta_moid",
    "etiqueta_pha",
    "etiqueta_pha_oficial",
    "etiqueta_proxy",
    "aplicar_etiquetas",
    "descripcion_etiqueta",
]

#: Nombre canónico -> (columna del dataframe, que definicion usa, prevalencia
#: tipica en el catalogo del proyecto). Se usa para rotular figuras y tablas.
ETIQUETAS = {
    "MOID_LE_005": {
        "columna": "et_moid_le_005",
        "definicion": f"MOID <= {UMBRAL_MOID} au",
        "nota": "Condición geométrica aislada. NO es un PHA: le falta el criterio de tamaño.",
    },
    "PHA": {
        "columna": "et_pha",
        "definicion": f"MOID <= {UMBRAL_MOID} au Y H <= {UMBRAL_H} mag",
        "nota": "Definición oficial de PHA (CNEOS/JPL), reconstruida desde el catálogo.",
    },
    "PHA_OFICIAL": {
        # La columna se llama `et_pha_oficial` y no `pha` porque es la que crea
        # `aplicar_etiquetas()`. Apuntar aqui a `pha` hacia que `--etiqueta pha`
        # buscas una columna que el dataframe de objetos no tiene, y el error
        # aparecia como un KeyError tardio, en la primera fila del CV.
        "columna": "et_pha_oficial",
        "definicion": "flag `pha` de la SBDB",
        "nota": "Ground truth declarado por JPL. Es lo que permite medir el techo de exactitud.",
    },
    "PROXY_OBSERVACIONAL": {
        "columna": "et_proxy",
        "definicion": f"H <= {UMBRAL_H} mag Y distnom_min <= {UMBRAL_MOID} au",
        "nota": "Sustitución de medida: usa solo lo registrado en aproximaciones cercanas.",
    },
}


def nombre_etiqueta(clave):
    """Texto para legends, títulos de figura y cabeceras de tabla.

    Se antepone siempre a la métrica: `F2@PHA (prev. 6.0%)` y no `F2`, que es lo
    que permitirá comparar en el futuro dos F2 sobre etiquetas distintas.
    """
    meta = ETIQUETAS[clave]
    return f"{clave} ({meta['definicion']})"


def descripcion_etiqueta(clave):
    """Aclaración de una línea sobre qué significa (y qué no) la etiqueta."""
    return ETIQUETAS[clave]["nota"]


def etiqueta_moid(moid, umbral=UMBRAL_MOID):
    """`MOID <= umbral`. Etiqueta continua de su partición binaria.

    Un MOID sin dato da NaN, no 0. `NaN <= 0.05` es `False` en numpy, asi que la
    comparacion directa etiquetaba como "no cumple" a los objetos de los que no se
    conoce la orbita, y el modelo aprendia que "sin dato" significa "no PHA".
    """
    moid = pd.Series(moid).astype(float)
    return pd.array((moid <= umbral).where(moid.notna()), dtype="Int64")


def etiqueta_pha(moid, h, umbral_moid=UMBRAL_MOID, umbral_h=UMBRAL_H):
    """Definición oficial de PHA: la condición geométrica Y la de tamaño.

    Tambien propaga NaN en cualquiera de los dos inputs, por el mismo motivo que
    `etiqueta_moid`.
    """
    moid = pd.Series(moid).astype(float)
    h = pd.Series(h).astype(float)
    cumple = ((moid <= umbral_moid) & (h <= umbral_h)).where(moid.notna() & h.notna())
    return pd.array(cumple, dtype="Int64")


def etiqueta_pha_oficial(flag_pha):
    """El flag `pha` de la SBDB, a entero. NaN se propaga como NaN."""
    return pd_to_nullable_int(flag_pha)


def etiqueta_proxy(h, distnom_min, umbral_moid=UMBRAL_MOID, umbral_h=UMBRAL_H):
    """Sustitución de medida: PHA a partir solo de lo registrado en la CAD.

    Ojo con el nombre: no es un PHA, es el resultado de aplicar los dos umbrales
    de la definición oficial a las cantidades observadas en los acercamientos.
    Su precisión alta es casi geométrica, no predictiva. Propaga NaN igual que
    las otras dos.
    """
    h = pd.Series(h).astype(float)
    distnom_min = pd.Series(distnom_min).astype(float)
    cumple = ((h <= umbral_h) & (distnom_min <= umbral_moid)).where(
        h.notna() & distnom_min.notna())
    return pd.array(cumple, dtype="Int64")


def aplicar_etiquetas(obj, moid_col="moid", h_col="H_sbdb", h_obs_col="H_obs",
                      dist_col="distnom_min", flag_col="pha"):
    """Añade al dataframe de objetos las cuatro etiquetas con nombre propio.

    `H_sbdb` se usa para la definición de PHA y no `H_obs` a propósito: la columna
    H de la CAD API es el mismo valor de catálogo que trae la SBDB (se verificó que
    coinciden en 339 316 de 340 135 filas), así que son indistinguibles, pero la
    de la SBDB deja claro de dónde sale el número.
    """
    import pandas as pd

    df = obj.copy()
    df["et_moid_le_005"] = etiqueta_moid(df[moid_col])
    df["et_pha"] = etiqueta_pha(df[moid_col], df[h_col])
    df["et_proxy"] = etiqueta_proxy(df[h_obs_col], df[dist_col])
    if flag_col in df.columns:
        df["et_pha_oficial"] = pd.to_numeric(df[flag_col], errors="coerce")
    return df


def pd_to_nullable_int(serie):
    """Como `pd.to_numeric` pero devolviendo enteros con hueco en vez de float."""
    return pd.to_numeric(serie, errors="coerce").astype("Int64")
