"""Constantes físicas, umbrales de la definición PHA y rutas del proyecto."""

import math
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(RAIZ, "data")
DIR_FIGURAS = os.path.join(RAIZ, "results", "figures")
RUTA_CAD = os.path.join(DIR_DATOS, "close_approaches.csv")
RUTA_SBDB = os.path.join(DIR_DATOS, "sbdb_neo.csv")

# Definición oficial de PHA: MOID <= 0.05 au Y H <= 22 mag (CNEOS FAQ, JPL/NASA)
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

# Semilla única del proyecto: PCA, K-Means y clasificación supervisada.
RANDOM_STATE = 20

# Tres cantidades independientes del bloque exploratorio. Se excluyen
# Diameter(km) (imputado desde H) y V relative (deducible de v_inf y la
# distancia) porque son funciones deterministas de las demás.
FEATURES_EXPLORATORIAS = ["CA DistanceNominal (au)", "V infinity(km/s)", "H(mag)"]

# Conjuntos de features de la clasificación supervisada (nivel objeto).
FEAT = {
    "kin+size": ["distnom_min", "vinf_max", "H_obs", "n_appro"],
    "kin-only": ["distnom_min", "vinf_max", "n_appro"],
    "size-only": ["H_obs"],
}
