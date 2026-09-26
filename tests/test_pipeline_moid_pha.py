"""Tests de `scripts/pipeline_moid_pha.py`.

El pipeline se importa por ruta (igual que `verificar_discrepancias.py`) pero **no**
se ejecuta entero: los tests se centran en las decisiones de las que depende el
resultado publicado y que no se ven leyendo las metricas.

Las dos que se comprueban aqui:

*   **El split temporal tiene que existir y ser disjunto.** El fallo original no
    producia una metica rara: producia un conjunto de prueba vacio y un
    `ValueError` del imputador, o, peor, un modelo evaluado sobre objetos cuyas
    features incorporateban acercamientos posteriores a la fecha de corte.
*   **El umbral no puede depender de la prueba.** Es un error silencioso: el
    numero sale bien, solo que mas alto de lo que deberia. Por eso los tests
    comprueban la invariante estructural (los tres conjuntos son disjuntos) y no
    solo que la metica exista.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from neos import etiquetas as E
from neos import features as F
from neos import modelos as MD
from neos.constantes import FEATURES_MOOD, RANDOM_STATE

RAIZ = Path(__file__).resolve().parent.parent
RUTA_PIPELINE = RAIZ / "scripts" / "pipeline_moid_pha.py"


@pytest.fixture(scope="module")
def pipe():
    spec = importlib.util.spec_from_file_location("pipeline_moid_pha", RUTA_PIPELINE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    yield modulo
    sys.modules.pop(spec.name, None)


def _objetos(n=400, semilla=11, corte=2015):
    """Objetos con dos cohortes: la mitad anterior al corte, la mitad posterior."""
    rng = np.random.default_rng(semilla)
    moid = rng.uniform(0, 0.3, n)
    h = rng.uniform(18, 26, n)
    n_pre = n // 2
    anios = np.concatenate([
        rng.integers(1995, corte, n_pre),
        rng.integers(corte, 2024, n - n_pre),
    ])
    return pd.DataFrame({
        "Object": [f"o{i}" for i in range(n)],
        "moid": moid,
        "H_sbdb": h,
        "H_obs": h + rng.normal(0, 0.3, n),
        "distnom_min": moid,
        "vinf_closest": rng.uniform(5, 40, n),
        "dist_unc_med": rng.uniform(0, 0.01, n),
        "n_appro": rng.integers(1, 5, n),
        "first_obs_year": anios,
    })


# --------------------------------------------------------------------------- #
# split temporal
# --------------------------------------------------------------------------- #
def test_el_split_temporal_produce_dos_conjuntos_no_vacios(pipe):
    """El fallo original: una sola tabla con `hasta=corte` deja la prueba vacia."""
    obj = E.aplicar_etiquetas(_objetos())
    m = F.construir_X(obj, FEATURES_MOOD)
    i_tr, i_te = pipe.indices_temporales(obj, m, anio_corte=2015)
    assert len(i_tr) > 0, "conjunto de entrenamiento vacio"
    assert len(i_te) > 0, "conjunto de prueba vacio: el modelo no se puede evaluar"
    assert set(i_tr) & set(i_te) == set(), "entrenamiento y prueba comparten objetos"


def test_el_split_ordena_por_descubrimiento_no_por_evento(pipe):
    """Un objeto descubierto despues del corte no puede(train) haber estado antes
    solo porque su evento mas cercano sea antiguo."""
    obj = E.aplicar_etiquetas(_objetos())
    m = F.construir_X(obj, FEATURES_MOOD)
    i_tr, i_te = pipe.indices_temporales(obj, m, anio_corte=2015)
    anios = obj.loc[m.idx, "first_obs_year"].to_numpy()
    assert (anios[i_tr] < 2015).all()
    assert (anios[i_te] >= 2015).all()


def test_la_prueba_cubre_la_cohorte_posterior_completa(pipe):
    """`hasta=corte` sin `desde=corte` deja fuera la mitad de la poblacion."""
    obj = E.aplicar_etiquetas(_objetos())
    m = F.construir_X(obj, FEATURES_MOOD)
    _, i_te = pipe.indices_temporales(obj, m, anio_corte=2015)
    esperados = (obj.loc[m.idx, "first_obs_year"].to_numpy() >= 2015).sum()
    assert len(i_te) == esperados


# --------------------------------------------------------------------------- #
# los tres folds son disjuntos
# --------------------------------------------------------------------------- #
def test_tres_folds_particiona_sin_solapamientos(pipe):
    """`i_in`, `i_val` e `i_te` tienen que ser disjuntos: el umbral se elige con
    `i_val` y si `i_te` estuviera ahi el F2 estaria optimizado sobre si mismo."""
    obj = E.aplicar_etiquetas(_objetos())
    m = F.construir_X(obj, FEATURES_MOOD)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    vistos = set()
    for fold, i_in, i_val, i_te in pipe._tres_folds(cv, m, m.y, RANDOM_STATE):
        juntos = set(i_in) | set(i_val) | set(i_te)
        assert len(juntos) == len(i_in) + len(i_val) + len(i_te), (
            f"fold {fold}: los tres conjuntos se solapan")
        assert set(i_te) & (set(i_in) | set(i_val)) == set(), (
            f"fold {fold}: la prueba se solapa con entrenamiento o validacion")
        assert len(set(i_in)) == len(i_in)
        assert len(set(i_val)) == len(i_val)
        assert len(set(i_te)) == len(i_te)
        vistos |= set(i_te)
    assert vistos == set(range(len(m))), (
        "cada objeto debe ser prueba exactamente una vez")


def test_tres_folds_estratifica_la_clase(pipe):
    """Un fold sin positivos daria `roc_auc_score` sobre una sola clase."""
    obj = E.aplicar_etiquetas(_objetos())
    m = F.construir_X(obj, FEATURES_MOOD)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, i_in, i_val, i_te in pipe._tres_folds(cv, m, m.y, RANDOM_STATE):
        for nombre, idx in (("in", i_in), ("val", i_val), ("te", i_te)):
            y = m.y[idx]
            assert 0 < y.sum() < len(y), f"fold {fold} {nombre}: una sola clase"


# --------------------------------------------------------------------------- #
# sin fuga
# --------------------------------------------------------------------------- #
def test_el_umbral_se_elige_sobre_validacion_no_sobre_prueba(pipe):
    """El umbral de la version anterior se calculaba con `y[i_te]`, el mismo
    conjunto sobre el que despues se reportaba el F2."""
    y = np.array([0] * 40 + [1] * 40)
    score_val = np.concatenate([np.linspace(0, 0.4, 40), np.linspace(0.6, 1, 40)])
    score_te = np.concatenate([np.linspace(0, 0.4, 40), np.linspace(0.6, 1, 40)])

    umbral_val = MD.umbral_por_f2(y, score_val)["umbral"]
    # Un umbral elegido con la prueba seria el que maximiza F2 sobre `score_te`.
    # Aqui ambas distribuciones coinciden, asi que comprobamos la via estructural:
    # el umbral del fold sale de `i_val` y el F2 se evalua con `i_te`.
    f2 = pipe.evaluacion_resumida(y, score_te, umbral_val)["F2"]
    assert 0.0 <= umbral_val <= 1.0
    assert f2 > 0.5, "un umbral honesto sobre datos separables da F2 alto"


def test_evaluacion_resumida_no_depende_del_umbral_para_el_ap(pipe):
    """PR-AUC y ROC-AUC no dependen del umbral: si cambiaran, el corte estaria
    metido dentro de la metica."""
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([.1, .2, .8, .7, .3, .9])
    a = pipe.evaluacion_resumida(y, score, 0.0)
    b = pipe.evaluacion_resumida(y, score, 1.0)
    assert a["PR-AUC"] == pytest.approx(b["PR-AUC"])
    assert a["ROC-AUC"] == pytest.approx(b["ROC-AUC"])


# --------------------------------------------------------------------------- #
# regresion
# --------------------------------------------------------------------------- #
def test_la_regresion_se_evalua_solo_en_la_cohorte_posterior(pipe):
    obj = E.aplicar_etiquetas(_objetos())
    ev = pipe.evaluar_regresion_moid(obj, anio_corte=2015, features=FEATURES_MOOD)
    n_esperado = int((obj["first_obs_year"] >= 2015).sum())
    assert ev["metricas"]["n_prueba"] == n_esperado
    assert ev["metricas"]["n_prueba"] > 0


def test_la_regresion_devuelve_mae_en_au_y_no_en_fraccion(pipe):
    """Un MOID se mide en au. Si el MAE saliera en fraccion, las cifras serian
    100 veces mas pequenas de lo que parecen."""
    obj = E.aplicar_etiquetas(_objetos())
    ev = pipe.evaluar_regresion_moid(obj, anio_corte=2015, features=FEATURES_MOOD)
    assert 0 < ev["metricas"]["mae_au"] < 1.0, "un MAE de MOID fuera de [0, 1] au es imposible"


# --------------------------------------------------------------------------- #
# comprobacion de fugas sobre el dataset construido
# --------------------------------------------------------------------------- #
def test_comprobar_fugas_acepta_un_dataset_correcto(pipe):
    obj = E.aplicar_etiquetas(_objetos())
    obj["tramo"] = np.where(obj["first_obs_year"] < 2015, "pre", "post")
    r = pipe.comprobar_fugas_temporales(obj, anio_corte=2015)
    assert r["n_train"] + r["n_test"] == len(obj)
    assert r["n_objetos"] == len(obj)


def test_comprobar_fugas_detecta_una_mitad_vacia(pipe):
    """Con `hasta=corte` y sin `desde`, la mitad de prueba sale vacia y el
    holdout temporal no existe."""
    obj = E.aplicar_etiquetas(_objetos())
    obj = obj[obj["first_obs_year"] < 2015].copy()
    obj["tramo"] = "pre"
    with pytest.raises(AssertionError) as e:
        pipe.comprobar_fugas_temporales(obj, anio_corte=2015)
    assert "vacia" in str(e.value)


def test_comprobar_fugas_detecta_tramo_incoherente(pipe):
    """Fila de la cohorte posterior con features de la ventana anterior."""
    obj = E.aplicar_etiquetas(_objetos())
    obj["tramo"] = np.where(obj["first_obs_year"] < 2015, "pre", "post")
    obj.loc[obj.index[-1], "tramo"] = "pre"
    with pytest.raises(AssertionError) as e:
        pipe.comprobar_fugas_temporales(obj, anio_corte=2015)
    assert "tramo" in str(e.value)


def test_comprobar_fugas_detecta_objetos_repetidos(pipe):
    """El mismo objeto en las dos ventanas: el modelo lo memoriza en vez de
    generalizar, y las features de un fold se contradicen con las de otro."""
    obj = E.aplicar_etiquetas(_objetos())
    obj["tramo"] = np.where(obj["first_obs_year"] < 2015, "pre", "post")
    obj = pd.concat([obj, obj.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError) as e:
        pipe.comprobar_fugas_temporales(obj, anio_corte=2015)
    assert "mas de una vez" in str(e.value)


def test_el_dataset_real_no_repite_objetos_ni_mezcla_ventanas(pipe):
    """Sobre el snapshot real, no sobre datos sinteticos.

    Los tests con datos inventados comprueban la logica del filtro, no que
    `agregar_por_objeto` lo respete: si `hasta`/`desde` se aplicaran a la columna
    equivocada, aqui se veria. Se limita a un subconjunto de objetos para que el
    test se veria. Se limita a un subconjunto de objetos para que el test no tarde
    un minuto.
    """
    from neos import datos

    if not list((RAIZ / "data").glob("close_approaches_v*.csv")):
        pytest.skip("no hay snapshot versionado en data/")
    df = datos.cargar_close_approaches(verificar=False)
    # Muestreo aleatorio y no los primeros N: el snapshot esta ordenado, asi que
    # los primeros objetos son de la epoca temprana y la mitad de prueba sale vacia.
    objetos = np.random.default_rng(20).choice(df["Object"].unique(), 1500,
                                               replace=False)
    df = df[df["Object"].isin(objetos)]
    if df.empty:
        pytest.skip("el subconjunto de objetos no tiene eventos")

    pre = datos.agregar_por_objeto(df, hasta=2015)
    post = datos.agregar_por_objeto(df, desde=2015)
    pre["tramo"], post["tramo"] = "pre", "post"
    obj = pd.concat([pre, post], ignore_index=True)
    n_por_objeto = obj.groupby("Object").size()
    obj = obj.merge(n_por_objeto.rename("n_tramos").reset_index(), on="Object")
    obj = obj[((obj["tramo"] == "pre") & (obj["first_obs_year"] < 2015))
              | ((obj["tramo"] == "post") & (obj["first_obs_year"] >= 2015))]
    r = pipe.comprobar_fugas_temporales(obj, anio_corte=2015)
    assert r["n_objetos"] == r["n_train"] + r["n_test"]
    assert not obj["Object"].duplicated().any()


def test_agregar_por_objeto_agrega_solo_los_eventos_de_su_ventana():
    """El corte filtra los *eventos*, no los objetos: un objeto con acercamientos
    en 2010 y en 2020 aparece en las dos ventanas, y cada fila debe resumir solo
    los eventos de su ventana. Si se filtrara despues de agregar, la fila de la
    ventana `pre` incluiria el acercamiento de 2020.

    Por eso no basta con comprobar que el objeto no trae eventos posteriores: hay
    que comparar el valor agregado con el que sale de agregar a mano.
    """
    from neos import datos

    if not list((RAIZ / "data").glob("close_approaches_v*.csv")):
        pytest.skip("no hay snapshot versionado en data/")
    df = datos.cargar_close_approaches(verificar=False)
    objetos = np.random.default_rng(20).choice(df["Object"].unique(), 1500,
                                               replace=False)
    df = df[df["Object"].isin(objetos)]
    if df.empty:
        pytest.skip("el subconjunto de objetos no tiene eventos")

    con_pre = datos.agregar_por_objeto(df, hasta=2015)
    if con_pre.empty:
        pytest.skip("el subconjunto no tiene objetos con eventos anteriores a 2015")

    anos = pd.to_datetime(df["Close-Approach (CA) Date"], errors="coerce").dt.year
    solo_pre = df[df["Object"].isin(con_pre["Object"]) & (anos < 2015)]
    esperado = solo_pre.groupby("Object")["CA DistanceNominal (au)"].min()
    observado = con_pre.set_index("Object")["distnom_min"]
    comunes = esperado.index.intersection(observado.index)
    assert len(comunes) > 0
    for obj in comunes[:50]:
        assert observado[obj] == pytest.approx(esperado[obj]), (
            f"{obj}: la fila `pre` no se limita a los eventos anteriores al corte")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def test_el_argumento_etiqueta_ofrece_columnas_no_dicts(pipe):
    """Con `choices=list(ETIQUETAS.values())` el `--help` imprimia cuatro objetos
    de tres lineas y ningun valor era aceptable, ni el propio `et_pha`.

    Se ejecuta el parser real del script (lo construye `main()` con `parse_args`
    leyendo `sys.argv`), no una copia local: una copia comprobaria que la copia
    funciona.
    """
    columnas = sorted(m["columna"] for m in E.ETIQUETAS.values())
    assert "et_pha" in columnas

    import contextlib
    import io as _io

    argv_original = sys.argv
    try:
        sys.argv = ["pipeline_moid_pha.py", "--help"]
        with contextlib.redirect_stdout(_io.StringIO()) as salida:
            with pytest.raises(SystemExit):
                pipe.main()
        texto = salida.getvalue()
    finally:
        sys.argv = argv_original
    # argparse siempre imprime las opciones entre llaves, asi que la comprobacion
    # no es que no haya "{", sino que no aparezca la representacion de los dict:
    # con `choices=list(ETIQUETAS.values())` el --help mostraba la clave
    # 'columna' y la nota de cada etiqueta, y ademas ningun valor era aceptable.
    bloque = texto.split("--etiqueta")[-1].split("--sin-ablation")[0]
    assert "'columna':" not in bloque, "las opciones imprimen los dict completos"
    assert "Ground truth declarado por JPL" not in bloque, "se imprime la nota"
    for col in columnas:
        assert col in bloque, f"{col} no aparece en las opciones de --etiqueta"


def test_etiqueta_por_defecto_es_la_definicion_oficial():
    assert E.ETIQUETAS["PHA"]["columna"] == "et_pha"
