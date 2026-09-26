"""Tests de scripts/verificar_discrepancias.py.

Cada bloque del script se ejercita con un catalogo sintetico minimo en el que
se conoce de antemano el veredicto esperado: un catalogo "sano" (todas las
comprobaciones corregibles pasan) y catalogos degenerados que reproducen la
situacion censurada que el script debe detectar.
"""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import (diametro_desde_h, evento, frame_cad, frame_sano,
                      v_rel_desde_v_inf)


# --------------------------------------------------------------------------
# Constantes y utilidades de reporte
# --------------------------------------------------------------------------

def test_constantes_coinciden_con_la_definicion_oficial(vd):
    assert vd.UMBRAL_MOID == 0.05
    assert vd.UMBRAL_H == 22.0
    assert vd.ALBEDO_ASUMIDO == 0.14
    assert vd.FACTOR_H_D == pytest.approx(3552.0, abs=1.0)


def test_veredicto_ok_no_acumula_fallo(vd, capsys):
    vd.veredicto(True, "todo bien")
    assert vd.fallos == []
    assert "OK" in capsys.readouterr().out


def test_veredicto_fallido_se_acumula_con_su_mensaje(vd, capsys):
    vd.veredicto(False, "algo mal")
    vd.veredicto(False, "otra cosa mal")
    assert vd.fallos == ["algo mal", "otra cosa mal"]
    assert "FALLO" in capsys.readouterr().out


def test_bloque_e_info_solo_imprimen(vd, capsys):
    vd.bloque("X", "titulo del bloque")
    vd.info("un detalle")
    salida = capsys.readouterr().out
    assert "[X] titulo del bloque" in salida
    assert "INFO   un detalle" in salida
    assert vd.fallos == []


# --------------------------------------------------------------------------
# cargar()
# --------------------------------------------------------------------------

def test_cargar_sin_csv_aborta_con_instrucciones(vd):
    with pytest.raises(SystemExit) as exc:
        vd.cargar()
    assert "ProyectoNeoRework_data.ipynb" in str(exc.value)


def test_cargar_sin_columna_post_discovery_aborta(vd, escribir_cad):
    escribir_cad(frame_sano().drop(columns=["post_discovery"]))
    with pytest.raises(SystemExit) as exc:
        vd.cargar()
    assert "post_discovery" in str(exc.value)


def test_cargar_filtra_eventos_integrados_y_agrega_por_objeto(vd, escribir_cad):
    escribir_cad(frame_sano())
    df, obs, obj = vd.cargar()

    assert len(df) == 5
    assert len(obs) == 4  # el evento con post_discovery=0 queda fuera
    assert set(obj.columns) == {"Object", "distnom_min", "distmin_min", "H_obs",
                               "H_sbdb", "moid", "pha", "n_appro"}

    fila = obj.set_index("Object").loc["cercano-grande"]
    # 0.002 au / H=17.0 pertenecen al evento integrado: no deben ganar el min
    assert fila.distnom_min == pytest.approx(0.01)
    assert fila.H_obs == pytest.approx(18.0)
    assert fila.n_appro == 1


def test_cargar_agrega_el_minimo_entre_varios_eventos_observados(vd, escribir_cad):
    escribir_cad(frame_cad([
        evento("multiple", 0.20, 19.0, moid=0.11, pha=0),
        evento("multiple", 0.08, 18.5, moid=0.11, pha=0),
    ]))
    _, _, obj = vd.cargar()
    fila = obj.iloc[0]
    assert fila.distnom_min == pytest.approx(0.08)
    assert fila.H_obs == pytest.approx(18.5)
    assert fila.n_appro == 2


# --------------------------------------------------------------------------
# Bloque A: censura en el umbral PHA
# --------------------------------------------------------------------------

def test_a_censura_pasa_con_catalogo_no_censurado(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    df, _, obj = vd.cargar()
    vd.a_censura(df, obj)
    assert vd.fallos == []
    assert "FALLO" not in capsys.readouterr().out


def test_a_censura_detecta_las_cinco_senales_de_un_catalogo_censurado(
        vd, escribir_cad):
    # Todo dentro de 0.05 au y con PHA == (H <= 22): es el catalogo que produce
    # la CAD API con su dist-max por defecto.
    escribir_cad(frame_cad([
        evento("a", 0.01, 18.0, moid=0.01, pha=1),
        evento("b", 0.02, 25.0, moid=0.02, pha=0),
        evento("c", 0.03, 20.0, moid=0.03, pha=1),
    ]))
    df, _, obj = vd.cargar()
    vd.a_censura(df, obj)
    assert len(vd.fallos) == 5


def test_a_censura_ignora_moid_ausente_al_medir_la_fraccion(vd, escribir_cad,
                                                           capsys):
    escribir_cad(frame_cad([
        evento("con-moid", 0.30, 19.0, moid=0.30, pha=0),
        evento("con-moid-2", 0.01, 18.0, moid=0.02, pha=1),
        evento("sin-moid", 0.02, 21.0, moid=None, pha=None),
    ]))
    df, _, obj = vd.cargar()
    vd.a_censura(df, obj)
    # 1 de los 2 objetos con MOID esta bajo el umbral: 50%, no 33%
    assert "MOID <= 0.05 au: 50.0%" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Bloque B: H es definicional
# --------------------------------------------------------------------------

def test_b_reporta_correlacion_y_maxima_diferencia_de_h(vd, capsys):
    df = frame_cad([
        evento("a", 0.10, 18.0, h_sbdb=18.0),
        evento("b", 0.20, 20.0, h_sbdb=20.5),
        evento("c", 0.30, 24.0, h_sbdb=24.0),
        evento("d", 0.40, 26.0, h_sbdb=None),  # se descarta por NaN
    ])
    vd.b_h_definicional(df)
    salida = capsys.readouterr().out
    assert "max|dif| = 0.5000 mag" in salida
    assert "corr(H_CAD, H_SBDB) = 0.99" in salida
    assert vd.fallos == []


# --------------------------------------------------------------------------
# Bloque C: features derivadas
# --------------------------------------------------------------------------

def test_c_detecta_diametro_imputado_desde_h(vd, capsys):
    df = frame_cad([
        evento("imputado", 0.10, 20.0),
        evento("medido", 0.20, 20.0, diameter=1.234),
    ])
    vd.c_features_redundantes(df)
    assert "imputado desde H en el 50.0%" in capsys.readouterr().out


def test_c_reporta_error_nulo_cuando_v_rel_sigue_la_relacion_fisica(vd, capsys):
    df = frame_cad([evento("a", 0.10, 20.0, v_inf=8.0),
                    evento("b", 0.02, 19.0, v_inf=15.0)])
    vd.c_features_redundantes(df)
    assert "error mediano 0.00000 km/s" in capsys.readouterr().out


def test_c_reporta_error_no_nulo_si_v_rel_es_inconsistente(vd, capsys):
    df = frame_cad([evento("a", 0.10, 20.0, v_inf=8.0,
                           v_rel=v_rel_desde_v_inf(8.0, 0.10) + 1.0)])
    vd.c_features_redundantes(df)
    assert "error mediano 1.00000 km/s" in capsys.readouterr().out
    assert vd.fallos == []


# --------------------------------------------------------------------------
# Bloque D: sesgo de muestreo
# --------------------------------------------------------------------------

def _sbdb(filas):
    return pd.DataFrame(filas, columns=["pdes", "pha", "moid", "H"])


def test_d_se_omite_si_falta_el_catalogo_sbdb(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    _, _, obj = vd.cargar()
    vd.d_sesgo_muestreo(obj)
    assert "omitido: falta data/sbdb_neo.csv" in capsys.readouterr().out
    assert vd.fallos == []


def test_d_pasa_cuando_el_catalogo_representa_a_la_poblacion(vd, escribir_cad,
                                                            capsys):
    escribir_cad(frame_cad([
        evento("1", 0.01, 18.0, moid=0.01, pha=1),
        evento("2", 0.30, 19.0, moid=0.30, pha=0),
    ]))
    _, _, obj = vd.cargar()
    _sbdb([
        ("1", "Y", 0.01, 18.0),
        ("2", "N", 0.30, 19.0),
        ("3", "N", 0.40, 25.0),
        ("4", "Y", 0.02, 17.0),
    ]).to_csv(vd.RUTA_SBDB, index=False)

    vd.d_sesgo_muestreo(obj)
    salida = capsys.readouterr().out
    assert "todos los NEOs (SBDB)" in salida
    assert "en el catálogo CAD" in salida
    assert vd.fallos == []  # 50% vs 50%: desviacion 0 pp


def test_d_falla_cuando_el_catalogo_esta_sesgado_en_moid(vd, escribir_cad):
    # Los 2 objetos del catalogo CAD estan todos bajo 0.05 au mientras la
    # poblacion solo lo esta en un 20%: desviacion de 80 pp.
    escribir_cad(frame_cad([
        evento("1", 0.01, 18.0, moid=0.01, pha=1),
        evento("2", 0.02, 17.0, moid=0.02, pha=1),
    ]))
    _, _, obj = vd.cargar()
    _sbdb([("1", "Y", 0.01, 18.0), ("2", "Y", 0.02, 17.0)]
          + [(str(i), "N", 0.40, 25.0) for i in range(3, 11)]
          ).to_csv(vd.RUTA_SBDB, index=False)

    vd.d_sesgo_muestreo(obj)
    assert len(vd.fallos) == 1
    assert "desviación en MOID<=0.05" in vd.fallos[0]


def test_d_convierte_valores_no_numericos_del_sbdb(vd, escribir_cad, capsys):
    escribir_cad(frame_cad([evento("1", 0.01, 18.0, moid=0.01, pha=1)]))
    _, _, obj = vd.cargar()
    _sbdb([("1", "Y", 0.01, 18.0),
           ("2", "N", "", ""),          # campos vacios de la SBDB
           ("3", "N", "n/a", "n/a")]).to_csv(vd.RUTA_SBDB, index=False)

    vd.d_sesgo_muestreo(obj)
    # Solo 1 de los 3 NEOs tiene MOID: la fraccion se calcula sobre ese
    assert "100.0%" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Bloques E a I
# --------------------------------------------------------------------------

def test_e_cuenta_objetos_con_cota_3sigma_degenerada(vd, escribir_cad, capsys):
    escribir_cad(frame_cad([
        evento("rasante", 0.02, 18.0, dist_min=1e-9, moid=0.02, pha=1),
        evento("normal", 0.30, 19.0, moid=0.30, pha=0),
    ]))
    _, _, obj = vd.cargar()
    vd.e_dist_min(obj)
    assert "min(dist_min) < 1e-5 au: 1" in capsys.readouterr().out
    assert vd.fallos == []


def test_f_reporta_la_fraccion_de_eventos_observados(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    df, _, _ = vd.cargar()
    vd.f_retroactivas(df)
    salida = capsys.readouterr().out
    assert "eventos observados: 4 de 5 (80.0%)" in salida
    assert vd.fallos == []


def test_g_mide_evolucion_secular_del_moid(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    _, _, obj = vd.cargar()
    vd.g_moid_epoca(obj)
    # solo "cercano-grande" tiene MOID (0.02) > distancia observada (0.01)
    assert "distancia observada mínima: 25.0%" in capsys.readouterr().out


def test_h_albedo_reporta_los_factores_de_escala_del_diametro(vd, capsys):
    vd.h_albedo()
    salida = capsys.readouterr().out
    assert f"{math.sqrt(0.14 / 0.030):.2f}x" in salida
    assert f"{math.sqrt(0.14 / 0.168):.2f}x" in salida
    assert vd.fallos == []


def test_i_flag_mide_el_techo_de_exactitud(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    _, _, obj = vd.cargar()
    vd.i_flag(obj)
    assert "vs flag pha: 100.00%" in capsys.readouterr().out


def test_i_flag_detecta_objetos_que_la_regla_no_reproduce(vd, escribir_cad,
                                                         capsys):
    escribir_cad(frame_cad([
        evento("coincide", 0.01, 18.0, moid=0.01, pha=1),
        # H y MOID cumplen la regla pero el flag oficial dice que no es PHA
        evento("discrepa", 0.02, 19.0, moid=0.02, pha=0),
    ]))
    _, _, obj = vd.cargar()
    vd.i_flag(obj)
    assert "vs flag pha: 50.00%" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------

def test_main_termina_sin_error_con_un_catalogo_sano(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    vd.main()
    salida = capsys.readouterr().out
    assert "Todas las comprobaciones corregibles pasan." in salida
    assert "Dataset: 5 eventos (4 observados), 4 objetos" in salida


def test_main_sale_con_codigo_1_y_lista_los_fallos(vd, escribir_cad, capsys):
    escribir_cad(frame_cad([
        evento("a", 0.01, 18.0, moid=0.01, pha=1),
        evento("b", 0.02, 25.0, moid=0.02, pha=0),
    ]))
    with pytest.raises(SystemExit) as exc:
        vd.main()
    assert exc.value.code == 1
    salida = capsys.readouterr().out
    assert "COMPROBACIÓN(ES) FALLIDA(S)" in salida
    assert len(vd.fallos) == 5


def test_main_recorre_todos_los_bloques(vd, escribir_cad, capsys):
    escribir_cad(frame_sano())
    vd.main()
    salida = capsys.readouterr().out
    for clave in "ABCDEFGHI":
        assert f"[{clave}]" in salida


def test_diametro_imputado_usa_el_mismo_factor_que_el_script(vd):
    assert diametro_desde_h(20.0) == pytest.approx(
        vd.FACTOR_H_D * 10 ** (-0.2 * 20.0))
    assert np.isclose(vd.FACTOR_H_D, 1329 / math.sqrt(0.14))
