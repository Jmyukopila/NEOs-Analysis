"""Constantes físicas, umbrales de la definición PHA, rutas y semillas del proyecto.

Este módulo es la única fuente de verdad para umbrales, rutas y semillas. Antes
cada script y cada notebook mantenía su propia copia, y divergieron (había tres
umbrales de decisión distintos y tres semillas distintas en el repositorio).
"""

import math
import os

# --------------------------------------------------------------------------- #
# Rutas. Ancladas a la raíz del repo, de modo que el directorio de trabajo
# actual deja de importar: los scripts se ejecutan igual desde la raíz, desde
# scripts/ o desde un notebook.
# --------------------------------------------------------------------------- #
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(RAIZ, "data")
DIR_RESULTADOS = os.path.join(RAIZ, "results")
DIR_FIGURAS = os.path.join(DIR_RESULTADOS, "figures")
DIR_TABLAS = os.path.join(DIR_RESULTADOS, "tables")

RUTA_CAD = os.path.join(DIR_DATOS, "close_approaches.csv")
RUTA_SBDB = os.path.join(DIR_DATOS, "sbdb_neo.csv")
RUTA_SNAPSHOT_INFO = os.path.join(DIR_DATOS, "snapshot_info.json")

# --------------------------------------------------------------------------- #
# Definición oficial de PHA (CNEOS/JPL): MOID <= 0.05 au Y H <= 22 mag.
# --------------------------------------------------------------------------- #
UMBRAL_MOID = 0.05  # au
UMBRAL_H = 22.0  # mag

# Relación estándar H-albedo-diámetro D = 1329·10^(-0.2H)/sqrt(p_V)
ALBEDO_ASUMIDO = 0.14
FACTOR_H_D = 1329 / math.sqrt(ALBEDO_ASUMIDO)  # ≈ 3552

AU_KM = 1.495978707e8  # km por unidad astronómica
MU_TIERRA = 3.986004418e5  # km^3/s^2, parámetro gravitacional terrestre

# dist-max de la CAD API. Su valor por defecto (0.05) es exactamente el umbral de
# distancia de la definición PHA: dejarlo implícito censura la muestra en el
# umbral de la propia etiqueta. 0.5 au es el máximo que sirve JPL.
DIST_MAX_AU = 0.5

# --------------------------------------------------------------------------- #
# Semillas globales.
#
# RANDOM_STATE es la semilla canónica del proyecto: la que se usa para el PCA, el
# K-Means y cualquier ajuste que no se repita. SEMILLAS es la lista con la que se
# repiten los experimentos para obtener intervalos de confianza; cada elemento
# genera un reparto de folds distinto y por tanto una estimación independiente de
# la métrica. La varianza entre semillas ES la incertidumbre que antes se reportaba
# como "±std de folds", que no lo era (los folds de una misma repetición comparten
# datos de entrenamiento).
#
# Cualquier experimento que necesite repetirse debe usar generar_semillas() y
# registrar la lista completa en su salida, para que el resultado sea auditable.
# --------------------------------------------------------------------------- #
RANDOM_STATE = 20
N_SEMILLAS = 10
SEMILLAS = tuple(RANDOM_STATE + i for i in range(N_SEMILLAS))

# --------------------------------------------------------------------------- #
# Cortes temporales.
#
# El holdout temporal se definía en tres sitios distintos y en dos direcciones
# opuestas (entrenar en el pasado / evaluar en el futuro, y al revés). Aquí hay
# una sola definición: se entrena con lo descubierto hasta ANIO_CORTE - 1 y se
# evalúa con lo descubierto desde ANIO_CORTE.
# --------------------------------------------------------------------------- #
ANIO_CORTE = 2015
ANIO_PRUBA_HISTORICA = 2000  # solo para el análisis de extrapolación documentado

# --------------------------------------------------------------------------- #
# Features.
# --------------------------------------------------------------------------- #
# Tres cantidades independientes del bloque exploratorio. Se excluyen
# Diameter(km) (imputado desde H) y V relative (deducible de v_inf y la
# distancia) porque son funciones deterministas de las demás.
FEATURES_EXPLORATORIAS = ["CA DistanceNominal (au)", "V infinity(km/s)", "H(mag)"]

# Conjuntos de features de la clasificación supervisada (nivel objeto).
#
# `kin+size`, `kin-only` y `size-only` son los tres que ya usaba el proyecto y hay
# que conservarlos para que las cifras históricas sigan siendo comparables. Los
# restantes son la ablación que faltaba y que decide si la conclusión "la
# cinemática aporta señal" se sostiene:
#
#   kin-puro      kinemática sin `n_appro` (exposición observacional)
#   size+n_appro  tamaño + exposición, para aislar el aporte de `n_appro`
#   sin-dist      todo MENOS la distancia observada: si no supera a `proxy-1feat`
#                 no hay señal cinemática independiente de la distancia
#   proxy-1feat   la regla de un solo umbral, que es el baseline real
FEAT = {
    "kin+size": ["distnom_min", "vinf_closest", "H_obs", "n_appro"],
    "kin-only": ["distnom_min", "vinf_closest", "n_appro"],
    "size-only": ["H_obs"],
    "kin-puro": ["distnom_min", "vinf_closest"],
    "size+n_appro": ["H_obs", "n_appro"],
    "sin-dist": ["vinf_closest", "H_obs", "n_appro", "dist_unc_med"],
    "proxy-1feat": ["distnom_min"],
}

# Conjunto principal del pipeline de predicción de MOID -> PHA. Sin `n_appro`:
# el número de aproximaciones registradas es exposición del sondeo, no física
# del objeto, y su mediana cae de 2 a 1 entre cohortes, así que incluirlo hace que
# el holdout temporal mida un cambio del muestreo y no del objeto.
FEATURES_MOOD = ["distnom_min", "vinf_closest", "H_obs", "dist_unc_med"]

# Tabla de equivalencias entre el nombre interno y el de la API, para las figuras
# y los ejes. Los nombres de las columnas del CSV se mantienen tal cual para no
# romper la trazabilidad con el snapshot congelado.
ETIQUETAS_FEATURES = {
    "distnom_min": "distancia nominal mínima (au)",
    "distmin_min": "distancia mínima 3σ (au)",
    "vinf_closest": "v∞ en el acercamiento más cercano (km/s)",
    "vinf_max": "máx. v∞ (km/s)",
    "H_obs": "H de catálogo (mag)",
    "n_appro": "nº de acercamientos registrados",
    "dist_unc_med": "incertidumbre orbital mediana (au)",
}

# Tolerancias de las comprobaciones de integridad (docs/05).
TOL_DIST_MAX = 0.06  # au: la muestra debe llegar más allá del umbral de 0.05
TOL_FRAC_CENSURA = 0.95  # fracción máxima admisible para considerarla censurada
TOL_DESVIACION_MOID_PP = 25.0  # pp de desviación tolerada en el bloque D
