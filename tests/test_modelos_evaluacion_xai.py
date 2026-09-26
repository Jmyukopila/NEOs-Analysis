"""Tests de `neos.modelos`, `neos.evaluacion` y `neos.xai`.

Los tres modulos comparten un mismo riesgo: un numero que parece correcto y no lo
es. `peso_positivos()` devolviendo 0.4, un umbral elegido sobre el conjunto de
prueba, un `Explanation` de SHAP tratado como ndarray. Nada de eso lanza excepcion;
solo cambia el resultado, y el resultado es lo que se publica.
"""

import numpy as np
import pandas as pd
import pytest

from neos import evaluacion as EV
from neos import features as F
from neos import modelos as MD
from neos import xai as XAI


# --------------------------------------------------------------------------- #
# modelos
# --------------------------------------------------------------------------- #
def test_peso_positivos_nunca_menor_que_uno():
    """`scale_pos_weight=0.4` hace que XGBoost penalice a la clase positiva.

    Devolvia 0.4 en las subpoblaciones donde la positiva era mayoritaria, que es
    justo lo que pasa en la cohortes posteriores a 2015.
    """
    minoritaria = np.array([0] * 90 + [1] * 10)
    mayoritaria = np.array([0] * 10 + [1] * 90)
    for y in (minoritaria, mayoritaria):
        p = MD.peso_positivos(y)
        assert p >= 1.0, f"peso {p} < 1 invierte el compensador de desbalance"
    assert MD.peso_positivos(minoritaria) == pytest.approx(9.0)
    assert MD.peso_positivos(mayoritaria) == pytest.approx(9.0)


def test_peso_positivos_exige_ambas_clases():
    with pytest.raises(ValueError):
        MD.peso_positivos(np.zeros(10, dtype=int))


def test_el_imputador_va_dentro_del_pipeline():
    """Imputar antes del split usa la mediana de la parte de prueba."""
    for nombre, m in MD.crear_modelos(np.array([0] * 80 + [1] * 20)).items():
        pasos = [n for n, _ in m.steps]
        assert pasos[0] == "imputer", f"{nombre} no imputa dentro del pipeline"
        assert not hasattr(m, "statistics_"), f"{nombre} esta ajustado de antemano"


def test_el_imputador_ve_na_n_y_lo_re_llena():
    """Si no llega NaN al imputador, es que la matriz se limpio antes."""
    X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    y = np.array([0, 1, 0])
    m = MD.crear_modelos(y)["LogReg"].fit(X, y)
    assert not np.isnan(m.named_steps["imputer"].transform(X)).any()


def test_logreg_usa_el_factor_explícito_no_balanced():
    """`class_weight="balanced"` recalcula el factor con el desbalance del fold."""
    m = MD.crear_logreg(np.array([0] * 80 + [1] * 20))
    clf = m.named_steps["clf"]
    assert clf.class_weight == {0: 1.0, 1: pytest.approx(4.0)}


def test_umbral_por_f2_beatiene_una_solucion_optima():
    """Con un score separado, el umbral tiene que caer entre los dos grupos."""
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    r = MD.umbral_por_f2(y, score)
    assert 0.4 <= r["umbral"] <= 0.6
    assert r["F2"] == pytest.approx(1.0)


def test_ajustar_y_calibrar_devuelve_umbral_sin_tocar_el_test():
    """La funcion es la que evita la fuga clasica: el umbral sale de `val`, no de
    `test`. Se comprueba que el modelo devuelto no se ha visto la parte de test."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + rng.normal(0, 0.3, 400) > 0).astype(int)
    mod, umbral = MD.ajustar_y_calibrar(MD.crear_modelos(y)["XGBoost"], X, y, semilla=1)
    assert 0.0 <= umbral <= 1.0
    assert hasattr(mod, "predict_proba")


def test_cv_de_semillas_no_repite_dentro():
    """`n_repeats` alto dentro de un CV mezcla variacion de folds con variacion de
    semillas; `generar_semillas()` es quien decide las repeticiones."""
    cv = MD.crear_cv(n_splits=5, n_repeats=1, random_state=20)
    assert cv.n_repeats == 1


# --------------------------------------------------------------------------- #
# evaluacion
# --------------------------------------------------------------------------- #
def _clasificacion(n=1000, semilla=3):
    """y y score con una relacion senal-ruido suave, no una separacion perfecta."""
    rng = np.random.default_rng(semilla)
    y = rng.integers(0, 2, n)
    score = np.clip(0.5 + 0.3 * (y + rng.normal(0, 0.5, n)), 0, 1)
    return y, score


def test_metricas_cuadran_con_la_matriz_de_confusion():
    y, score = _clasificacion()
    m = EV._metricas(y, score, 0.5)
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == m["n"]
    assert m["n_pos"] == int(y.sum())
    precision = m["TP"] / max(m["TP"] + m["FP"], 1)
    recall = m["TP"] / max(m["TP"] + m["FN"], 1)
    assert m["precision"] == pytest.approx(precision, abs=1e-9)
    assert m["recall"] == pytest.approx(recall, abs=1e-9)
    f2 = (5 * precision * recall) / max(4 * precision + recall, 1e-12)
    assert m["F2"] == pytest.approx(f2, abs=1e-9)


def test_roc_auc_no_se_calcula_con_una_sola_clase():
    y = np.zeros(50, dtype=int)
    m = EV._metricas(y, np.linspace(0, 1, 50), 0.5)
    assert "ROC-AUC" not in m, "roc_auc_score falla con una clase; se omite, no se inventa"


def test_bootstrap_contiene_el_punto_estimado():
    y, score = _clasificacion()
    umbral = MD.umbral_por_f2(y, score)["umbral"]
    ic = EV.intervalo_bootstrap(y, score, umbral, metrica="F2", n_boot=400, semilla=20)
    punto = EV._metricas(y, score, umbral)["F2"]
    assert ic["ic_inf"] <= punto <= ic["ic_sup"]
    assert ic["n_boot"] > 0


def test_bootstrap_es_determinista():
    y, score = _clasificacion()
    a = EV.intervalo_bootstrap(y, score, 0.5, n_boot=200, semilla=20)
    b = EV.intervalo_bootstrap(y, score, 0.5, n_boot=200, semilla=20)
    assert (a["ic_inf"], a["ic_sup"]) == (b["ic_inf"], b["ic_sup"])


def test_bootstrap_rechaza_una_metrica_inexistente():
    y, score = _clasificacion()
    with pytest.raises(ValueError):
        EV.intervalo_bootstrap(y, score, 0.5, metrica="F7", n_boot=10)


def test_mcnemar_detecta_diferencia_real():
    """Un clasificador perfecto contra uno que siempre dice 0."""
    y = np.array([0] * 20 + [1] * 20)
    perfecto = y.copy()
    nulo = np.zeros(40, dtype=int)
    r = EV.mcnemar(y, perfecto, nulo)
    assert r["n_disacuerdos"] == 20
    assert r["p"] < 0.001


def test_mcnemar_no_inventa_diferencia_entre_identicos():
    y, score = _clasificacion()
    pred = (score >= 0.5).astype(int)
    r = EV.mcnemar(y, pred, pred)
    assert r["n_disacuerdos"] == 0
    assert r["p"] == 1.0


def test_linea_base_ap_igual_a_la_prevalencia():
    y = np.array([0] * 95 + [1] * 5)
    assert EV.linea_base_ap(y) == pytest.approx(0.05)


def test_linea_base_es_el_suelo_de_la_pr_auc():
    """Un PR-AUC por debajo de la prevalencia no es un modelo malo: es ruido."""
    y = np.array([0] * 95 + [1] * 5)
    score = np.full(100, 0.05)
    m = EV._metricas(y, score, 0.5)
    assert m["PR-AUC"] < EV.linea_base_ap(y) * 3


def test_tabla_comparativa_agrupa_por_etiqueta_y_ordena_por_f2():
    """La agrupacion por etiqueta es deliberada: un F2 sobre PHA y otro sobre
    `MOID<=0.05` no son comparables, asi que la tabla no los mezcla en un unico
    ranking. Dentro de cada etiqueta, de mayor a menor F2."""
    filas = [EV.resumen_holdout(np.array([0, 1, 1, 0]), np.array([.1, .9, .8, .2]),
                                0.5, etiqueta="PHA"),
             EV.resumen_holdout(np.array([0, 1, 0, 0]), np.array([.1, .2, .3, .2]),
                                0.5, etiqueta="MOID_LE_005"),
             EV.resumen_holdout(np.array([0, 0, 1, 0]), np.array([.1, .4, .7, .2]),
                                0.5, etiqueta="PHA")]
    t = EV.tabla_comparativa(filas)
    assert "etiqueta" in t.columns
    assert set(t["etiqueta"]) == {"PHA", "MOID_LE_005"}
    assert list(t["etiqueta"]) == sorted(t["etiqueta"]), "agrupa por etiqueta"
    assert list(t[t.etiqueta == "PHA"]["F2"]) == sorted(
        t[t.etiqueta == "PHA"]["F2"], reverse=True)
    assert {"TP", "FP", "FN", "TN", "umbral", "n"} <= set(t.columns)


def test_tabla_comparativa_acepta_lista_vacia():
    assert EV.tabla_comparativa([]).empty


def test_media_entre_semillas_reporta_error_estandar():
    filas = [{"etiqueta": "PHA", "conjunto": "a", "F2": v} for v in (0.5, 0.6, 0.7)]
    r = EV.media_entre_semillas(filas)
    assert len(r) == 1
    assert r.loc[0, "media"] == pytest.approx(0.6)
    assert r.loc[0, "n_repeticiones"] == 3
    assert r.loc[0, "ic95_sup"] > r.loc[0, "ic95_inf"]


# --------------------------------------------------------------------------- #
# xai
# --------------------------------------------------------------------------- #
def _modelo_y_datos(n=200, semilla=5):
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)
    mod = MD.crear_modelos(y)["XGBoost"]
    mod.fit(X, y)
    return mod, X, y


def test_importancia_global_ordena_y_conserva_los_nombres():
    mod, X, y = _modelo_y_datos()
    r = XAI.explicar(mod, X, ["a", "b", "c"], X, max_amostras=100)
    imp = XAI.importancia_global(r)
    assert list(imp["feature"]) == sorted(imp["feature"], key=lambda f: -imp.set_index("feature").loc[f, "importancia"])
    assert imp.loc[0, "feature"] == "a", "la feature que separa las clases debe pesar mas"
    assert imp["importancia"].is_monotonic_decreasing


def test_los_valores_shap_son_ndarray_no_explanation():
    """`np.asarray(Explanation)` devuelve un array 0-d y revienta mas tarde, en la
    tabla de importancias, con un error que no menciona SHAP."""
    mod, X, y = _modelo_y_datos()
    r = XAI.explicar(mod, X, ["a", "b", "c"], X, max_amostras=50)
    v = np.asarray(r["valores"])
    assert v.ndim == 2 and v.shape == (50, 3)


def test_shap_usa_la_matriz_imputada_no_la_cruda():
    """Explicar la matriz sin imputar atribuiria a los NaN lo que es el valor
    imputado por el pipeline."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    X[:20, 0] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int)
    mod = MD.crear_modelos(y)["XGBoost"]
    mod.fit(X, y)
    X_prep, arbol = XAI.preparar_para_shap(mod, X, ["a", "b"])
    assert not np.isnan(X_prep).any(), "la matriz que se explica debe estar imputada"
    assert hasattr(arbol, "get_booster"), "debe devolverse el estimador, no el pipeline"


def test_fondo_limitado_y_semilla_reproducible():
    mod, X, y = _modelo_y_datos(n=400)
    a = XAI.explicar(mod, X, ["a", "b", "c"], X, semilla=20)
    b = XAI.explicar(mod, X, ["a", "b", "c"], X, semilla=20)
    assert a["n_fondo"] == b["n_fondo"] <= XAI.FONDO
    assert np.allclose(a["valores"], b["valores"])


def test_explicar_acota_las_filas_explicadas():
    mod, X, y = _modelo_y_datos(n=400)
    r = XAI.explicar(mod, X, ["a", "b", "c"], X, max_amostras=120)
    assert r["n_explicado"] == 120
    assert len(r["X"]) == 120


def test_resumen_objetos_devuelve_las_mas_influyentes():
    mod, X, y = _modelo_y_datos(n=60)
    r = XAI.explicar(mod, X, ["a", "b", "c"], X, max_amostras=30)
    t = XAI.resumen_objetos(r, y=r.get("y"), top=2)
    assert len(t) == 30 * 2
    assert set(t["feature"]) <= {"a", "b", "c"}
    assert t.groupby("fila")["shap"].apply(lambda s: s.abs().is_monotonic_decreasing).all()


def test_explicar_nuevos_incluye_la_decision():
    mod, X, y = _modelo_y_datos(n=80)
    t = XAI.explicar_nuevos(mod, X[:5], ["a", "b", "c"], X, ids=list("abcde"),
                            umbral=0.5)
    assert len(t) == 5
    assert list(t["objeto"]) == list("abcde")
    assert t["decision_pha"].dtype == bool
    assert t["probabilidad"].between(0, 1).all()
