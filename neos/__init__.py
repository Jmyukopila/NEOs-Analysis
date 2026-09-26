"""Utilidades compartidas del proyecto NEOs-Analysis.

Los notebooks (`data/`, `notebooks/`) y los scripts (`scripts/`) comparten
constantes, carga de datos, agregación por objeto, modelos y helpers de figuras.
Antes cada uno mantenía su propia copia; aquí viven una sola vez.

Los módulos son:

*   `constantes`: umbrales, rutas, semilla canónica y las features de cada conjunto.
*   `semillas`: fijación de semillas y derivados por componente.
*   `datos`: carga, verificación del snapshot y agregación por objeto.
*   `etiquetas`: las cuatro definiciones de etiqueta y por qué son distintas.
*   `features`: matriz de diseño, alias de columnas y auditoría de fugas.
*   `modelos`: imputación dentro del pipeline, pesos y calibración de umbral.
*   `evaluacion`: métricas, bootstrap, McNemar y línea base de PR-AUC.
*   `xai`: preparación de matrices e importancia SHAP.
*   `graficos`: figuras.

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

# Se importan aqui para que `from neos import datos` funcione sin escribir el
# submodulo, que es como los usan los notebooks. No se reexportan simbolos sueltos:
# el `__all__` de cada modulo es la referencia de lo que expone.
from neos import (  # noqa: F401  (importado para `from neos import <modulo>`)
    constantes,
    datos,
    etiquetas,
    evaluacion,
    features,
    graficos,
    modelos,
    semillas,
    xai,
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
    "constantes",
    "datos",
    "etiquetas",
    "evaluacion",
    "features",
    "graficos",
    "modelos",
    "semillas",
    "xai",
]
