"""Tests de `neos.constantes`, `neos.semillas`, `neos.etiquetas` y `neos.features`.

Cada test cubre un comportamiento que se rompió de verdad en algún momento del
repo, no la forma de la API. El mas repetido: el sistema de coordenadas. Las
columnas del dataframe y la posicion de una feature en la matriz de diseño estan
en el orden que decide el diccionario, no el que parece al leer, y basta con que
un script pase `features` en otro orden para que las metricas sean de otra
columna sin que nada falle.
"""

import numpy as np
import pandas as pd
import pytest

from neos import constantes as K
from neos import etiquetas as E
from neos import features as F
from neos import semillas as S


# --------------------------------------------------------------------------- #
# constantes
# --------------------------------------------------------------------------- #
def test_umbrales_de_pha_coinciden_con_la_definicion_oficial():
    """`MOID <= 0.05` sin el criterio de tamaño no es un PHA."""
    assert K.UMBRAL_MOID == 0.05
    assert K.UMBRAL_H == 22.0


def test_constantes_fisicas_validas():
    """`MU_TIERRA` en km^3/s^2. En m^3/s^2 el termino gravitatorio de la ecuacion
    de velocidad relativa sale 1e6 veces pequeno y `v_rel` queda dominado por
    `v_inf` en todos los casos, sin que nada falle."""
    assert K.AU_KM == pytest.approx(1.495978707e8, rel=1e-9)
    assert K.MU_TIERRA == pytest.approx(398600.4418, rel=1e-9)


def test_la_semilla_canonica_no_coincide_con_las_que_habia():
    """20 en unos sitios y 42 en otros hacia que los resultados no se comparaban."""
    assert K.RANDOM_STATE == 20
    assert K.SEMILLAS == tuple(range(K.RANDOM_STATE, K.RANDOM_STATE + K.N_SEMILLAS))
    assert len(K.SEMILLAS) == K.N_SEMILLAS


def test_toda_constante_que_los_scripts_usan_existe():
    """`predict_moid.py` importaba un nombre que no estaba aqui."""
    for nombre in ("ANIO_CORTE", "FEATURES_MOOD", "FEAT", "UMBRAL_MOID",
                   "UMBRAL_H", "RANDOM_STATE", "SEMILLAS", "N_SEMILLAS",
                   "ETIQUETAS_FEATURES", "MU_TIERRA", "AU_KM", "FACTOR_H_D",
                   "ANIO_PRUBA_HISTORICA"):
        assert hasattr(K, nombre), f"neos.constantes.{nombre} no existe"


def test_etiquetas_features_cubren_todas_las_features_de_los_conjuntos():
    """Una feature sin nombre legible sale como `feature_3` en las figuras."""
    usadas = {f for cols in K.FEAT.values() for f in cols} | set(K.FEATURES_MOOD)
    sin_nombre = usadas - set(K.ETIQUETAS_FEATURES)
    assert not sin_nombre, f"features sin nombre en ETIQUETAS_FEATURES: {sin_nombre}"


# --------------------------------------------------------------------------- #
# semillas
# --------------------------------------------------------------------------- #
def test_semillas_derivadas_distintas_por_componente():
    """Misma semilla base, generadores distintos: si coinciden, el muestreo de
    XGBoost y el split de sklearn quedan correlacionados."""
    d = S.fijar_semilla(20)
    assert d["global"] == 20
    valores = [d["sklearn"], d["xgboost"], d["kmeans"], d["shap"]]
    assert len(set(valores)) == len(valores)
    assert all(v > 20 for v in valores)


def test_fijar_semillas_todas_devuelve_las_derivadas_y_fija_el_generador():
    derivadas = S.fijar_semillas_todas(20, n_threads=1)
    assert derivadas["global"] == 20
    assert derivadas == S.fijar_semilla(20)  # mismo dict, no uno nuevo con otro valor
    a = np.random.random(3)
    S.fijar_semillas_todas(20, n_threads=1)
    b = np.random.random(3)
    assert np.array_equal(a, b)


def test_generar_semillas_es_contigua_y_reproducible():
    assert S.generar_semillas(3) == (20, 21, 22)
    assert S.generar_semillas(3) == S.generar_semillas(3)


# --------------------------------------------------------------------------- #
# etiquetas
# --------------------------------------------------------------------------- #
def test_et_pha_exige_los_dos_criterios():
    """El error que hace que el 59 % de los objetos se llamen PHA."""
    moid = pd.Series([0.01, 0.04, 0.06, 0.01])
    h = pd.Series([20.0, 22.0, 18.0, 23.0])
    r = E.etiqueta_pha(moid, h)
    assert list(r) == [1, 1, 0, 0], "MOID<=0.05 Y H<=22, no solo una de las dos"


def test_moid_le_005_no_es_pha():
    """Las dos etiquetas no pueden coincidir, o la confusion vuelve a propagarse."""
    moid = pd.Series([0.01, 0.06])
    h = pd.Series([21.0, 21.0])
    assert list(E.etiqueta_moid(moid)) == [1, 0]
    assert list(E.etiqueta_pha(moid, h)) == [1, 0]
    # Un objeto H>22 con MOID<=0.05: geometrica si, PHA no.
    moid2, h2 = pd.Series([0.01]), pd.Series([23.0])
    assert list(E.etiqueta_moid(moid2)) == [1]
    assert list(E.etiqueta_pha(moid2, h2)) == [0]


def test_proxy_usa_las_columnas_observadas():
    r = E.etiqueta_proxy(pd.Series([22.0, 22.0, 23.0]), pd.Series([0.04, 0.06, 0.04]))
    assert list(r) == [1, 0, 0]


def test_etiqueta_pha_oficial_conserva_nan():
    """`astype(int)` sobre una columna con huecos convierte los NaN en un numero
    enorme en vez de dejarlos como ausencia."""
    r = E.etiqueta_pha_oficial(pd.Series([1, 0, None]))
    assert r.isna().sum() == 1
    assert r.dropna().tolist() == [1, 0]


def test_aplicar_etiquetas_crea_todas_las_columnas_declaradas():
    """Cada etiqueta de ETIQUETAS debe existir tras aplicar: si el diccionario
    declara una columna que la funcion no crea, `--etiqueta` falla en runtime."""
    obj = pd.DataFrame({
        "moid": [0.01, 0.20, 0.01],
        "H_sbdb": [20.0, 21.0, 23.0],
        "H_obs": [20.0, 21.0, 23.0],
        "distnom_min": [0.02, 0.30, 0.02],
        "pha": [1, 0, 0],
    })
    out = E.aplicar_etiquetas(obj)
    for clave, meta in E.ETIQUETAS.items():
        assert meta["columna"] in out.columns, (
            f"ETIQUETAS[{clave!r}] declara la columna {meta['columna']!r} pero "
            "aplicar_etiquetas() no la crea")


def test_pha_oficial_apunta_a_la_columna_que_se_crea():
    """El bug: la clave apuntaba a `pha` y la funcion creaba `et_pha_oficial`."""
    assert E.ETIQUETAS["PHA_OFICIAL"]["columna"] == "et_pha_oficial"


def test_nombre_etiqueta_incluye_la_definicion():
    """`F2` a secas no dice sobre que etiqueta; el prefijo obliga a declararlo."""
    assert "0.05" in E.nombre_etiqueta("PHA")
    assert "22" in E.nombre_etiqueta("PHA")


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _objetos_sinteticos(n=60, semilla=7):
    """Objetos con MOID y H correlacionados con la etiqueta de PHA."""
    rng = np.random.default_rng(semilla)
    moid = rng.uniform(0.0, 0.3, n)
    h = rng.uniform(18.0, 26.0, n)
    return pd.DataFrame({
        "Object": [f"o{i}" for i in range(n)],
        "moid": moid,
        "H_sbdb": h,
        "H_obs": h + rng.normal(0, 0.3, n),
        "distnom_min": moid + rng.uniform(0, 0.02, n),
        "vinf_closest": rng.uniform(5, 40, n),
        "dist_unc_med": rng.uniform(0, 0.01, n),
        "n_appro": rng.integers(1, 5, n),
        "first_obs_year": rng.integers(1990, 2024, n),
        "pha": rng.integers(0, 2, n),
    })


def test_columnas_de_conserva_el_orden_de_features():
    """El orden de `FEATURES_MOOD` es el orden de las columnas de X. Si
    `columnas_de` usara un set, cada script usaria un orden distinto."""
    obj = _objetos_sinteticos()
    cols = F.columnas_de(K.FEATURES_MOOD, obj)
    assert cols == list(K.FEATURES_MOOD)


def test_columnas_de_acepta_nombres_amigables():
    """Los notebooks usan los nombres de la API; el pipeline usa los internos."""
    obj = _objetos_sinteticos()
    assert F.columna_de("distnom_min", obj) == "distnom_min"
    assert F.columnas_de(["distnom_min", "H_obs"], obj) == ["distnom_min", "H_obs"]


def test_construir_X_acepta_clave_o_columna_de_etiqueta():
    """Notebooks usan 'PHA' y scripts 'et_pha'; los dos tienen que funcionar."""
    obj = E.aplicar_etiquetas(_objetos_sinteticos())
    m1 = F.construir_X(obj, K.FEATURES_MOOD, etiqueta="PHA")
    m2 = F.construir_X(obj, K.FEATURES_MOOD, etiqueta="et_pha")
    assert m1.etiqueta == m2.etiqueta == "PHA"
    assert np.array_equal(m1.y, m2.y)


def test_etiqueta_desconocida_dice_cuales_hay():
    with pytest.raises(KeyError) as e:
        F.construir_X(_objetos_sinteticos(), K.FEATURES_MOOD, etiqueta="pha_oficial_2")
    assert "PHA" in str(e.value), "el error debe listar las etiquetas disponibles"


def test_construir_X_no_descarta_filas_por_falta_de_datos():
    """`dropna=True` eliminaba las filas antes de que el imputador del pipeline
    tuviera nada que hacer, y la imputacion no se podia ni medir ni auditar."""
    obj = E.aplicar_etiquetas(_objetos_sinteticos())
    obj.loc[obj.index[:10], "H_obs"] = np.nan
    m = F.construir_X(obj, K.FEATURES_MOOD)
    assert len(m) == len(obj), "con imputador en el pipeline no hay que descartar filas"
    assert np.isnan(m.X[:, K.FEATURES_MOOD.index("H_obs")]).any()


def test_construir_X_descarta_filas_sin_etiqueta_pero_no_por_falta_de_features():
    obj = _objetos_sinteticos()
    # El hueco va ANTES de aplicar las etiquetas: sin `moid` no hay `et_pha`, y una
    # fila sin `y` no tiene destino aunque el imputador pueda rellenar sus features.
    obj.loc[obj.index[:5], "moid"] = np.nan
    obj = E.aplicar_etiquetas(obj)
    obj.loc[obj.index[5:10], "H_obs"] = np.nan
    m = F.construir_X(obj, K.FEATURES_MOOD)
    assert len(m) == len(obj) - 5, "sin `moid` no hay etiqueta; sin `H_obs` si se imputa"


def test_matriz_y_es_ndarray_indexable_por_posicion():
    """Con una Serie, `y[i_tr]` con indices de numpy hacia un KeyError; con
    listas, pandas devuelve filas por etiqueta y el resultado es peor."""
    obj = E.aplicar_etiquetas(_objetos_sinteticos())
    m = F.construir_X(obj, K.FEATURES_MOOD)
    assert isinstance(m.y, np.ndarray)
    assert m.y[[0, 5, 9]].shape == (3,)


def test_matriz_reindexa_con_etiqueta_nueva():
    obj = E.aplicar_etiquetas(_objetos_sinteticos())
    m = F.construir_X(obj, K.FEATURES_MOOD, etiqueta="PROXY_OBSERVACIONAL")
    assert m.etiqueta == "PROXY_OBSERVACIONAL"
    m2 = m.con_etiqueta("MOID_LE_005", obj)
    assert np.array_equal(m2.y, obj.loc[m.idx, "et_moid_le_005"].to_numpy())


def test_nombres_amigables_no_son_identificadores():
    """Un eje con `distnom_min` obliga a mirar la tabla de equivalencias."""
    for f in K.FEATURES_MOOD:
        assert F.nombres_amigables(f) != f


def test_auditar_fugas_ordena_por_correlacion_con_el_anio():
    """Es la tabla que justifica excluir `n_appro`: sin el orden, la justificacion
    del pipeline no se puede escribir sola."""
    obj = E.aplicar_etiquetas(_objetos_sinteticos())
    inf = F.auditar_fugas(obj)
    assert "corr_anio" in inf.columns
    assert "n_appro" in set(inf["feature"])
    assert list(inf["corr_anio"].abs()) == sorted(inf["corr_anio"].abs(), reverse=True)


def test_comprobar_clase_avisa_si_la_clase_es_minuscula():
    obj = E.aplicar_etiquetas(_objetos_sinteticos(n=20))
    m = F.construir_X(obj, K.FEATURES_MOOD)
    with pytest.raises(ValueError) as e:
        F.comprobar_clase(m.X, np.zeros(len(m)), razon_min=0.002, minimo=200)
    assert str(e.value), "un recorte sin mensaje no se puede corregir"
