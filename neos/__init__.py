"""Utilidades compartidas del proyecto NEOs-Analysis.

Los notebooks (`data/`, `notebooks/`) y los scripts (`scripts/`) comparten
constantes, carga de datos, agregación por objeto, modelos y helpers de figuras.
Antes cada uno mantenía su propia copia; aquí viven una sola vez.

Uso desde un notebook, que se ejecuta con su propia carpeta como directorio de
trabajo:

    import os, sys
    sys.path.insert(0, os.path.abspath(".."))
    from neos import datos, graficos
"""

from neos.constantes import (
    ALBEDO_ASUMIDO,
    AU_KM,
    DIST_MAX_AU,
    FACTOR_H_D,
    FEATURES_EXPLORATORIAS,
    MU_TIERRA,
    RAIZ,
    RANDOM_STATE,
    RUTA_CAD,
    RUTA_SBDB,
    UMBRAL_H,
    UMBRAL_MOID,
)

__all__ = [
    "ALBEDO_ASUMIDO",
    "AU_KM",
    "DIST_MAX_AU",
    "FACTOR_H_D",
    "FEATURES_EXPLORATORIAS",
    "MU_TIERRA",
    "RAIZ",
    "RANDOM_STATE",
    "RUTA_CAD",
    "RUTA_SBDB",
    "UMBRAL_H",
    "UMBRAL_MOID",
    "datos",
    "graficos",
    "modelos",
]
