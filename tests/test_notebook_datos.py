"""Tests de las funciones auxiliares de data/ProyectoNeoRework_data.ipynb.

Las funciones se extraen del notebook (ver `conftest.funciones_notebook`) y se
ejecutan sin tocar la red: `descargar_cad` recibe un doble de `requests.get`.
"""

import os
import time

import pandas as pd
import pytest
import requests

from conftest import NB_DATOS, NB_ML, fuente_funcion


# --------------------------------------------------------------------------
# limpiar_fecha
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entrada, esperado", [
    ("2010-Jan-01 00:38 ±00:01", "2010-Jan-01 00:38"),
    ("2010-Jan-01 00:38", "2010-Jan-01 00:38"),
    ("  2010-Jan-01 00:38  ±< 00:01", "2010-Jan-01 00:38"),
    ("", ""),
])
def test_limpiar_fecha_elimina_el_sufijo_de_incertidumbre(nb_datos, entrada,
                                                          esperado):
    assert nb_datos["limpiar_fecha"](entrada) == esperado


def test_limpiar_fecha_deja_pasar_valores_no_texto(nb_datos):
    limpiar_fecha = nb_datos["limpiar_fecha"]
    assert limpiar_fecha(None) is None
    assert limpiar_fecha(42) == 42
    assert pd.isna(limpiar_fecha(float("nan")))


def test_limpiar_fecha_es_idempotente(nb_datos):
    limpiar_fecha = nb_datos["limpiar_fecha"]
    una = limpiar_fecha("2010-Jan-01 00:38 ±00:01")
    assert limpiar_fecha(una) == una


def test_limpiar_fecha_identica_en_los_dos_notebooks():
    """El notebook de ML duplica la funcion a proposito: deben coincidir."""
    assert fuente_funcion(NB_DATOS, "limpiar_fecha", sin_docstring=True) == \
        fuente_funcion(NB_ML, "limpiar_fecha", sin_docstring=True)


# --------------------------------------------------------------------------
# archivo_es_reciente
# --------------------------------------------------------------------------

def test_archivo_es_reciente_falso_si_no_existe(nb_datos, tmp_path):
    assert nb_datos["archivo_es_reciente"](str(tmp_path / "no-existe.csv")) is False


def test_archivo_es_reciente_verdadero_para_archivo_nuevo(nb_datos, tmp_path):
    ruta = tmp_path / "cache.csv"
    ruta.write_text("a\n")
    assert nb_datos["archivo_es_reciente"](str(ruta)) is True


def test_archivo_es_reciente_falso_pasado_el_plazo(nb_datos, tmp_path):
    ruta = tmp_path / "cache.csv"
    ruta.write_text("a\n")
    hace_31_dias = time.time() - 31 * 86400
    os.utime(ruta, (hace_31_dias, hace_31_dias))
    assert nb_datos["archivo_es_reciente"](str(ruta)) is False
    # con un plazo mas largo el mismo archivo sigue valiendo
    assert nb_datos["archivo_es_reciente"](str(ruta), dias_maximos=60) is True


def test_archivo_es_reciente_respeta_el_limite_exacto(nb_datos, tmp_path):
    ruta = tmp_path / "cache.csv"
    ruta.write_text("a\n")
    hace_10_dias = time.time() - 10 * 86400
    os.utime(ruta, (hace_10_dias, hace_10_dias))
    assert nb_datos["archivo_es_reciente"](str(ruta), dias_maximos=11) is True
    assert nb_datos["archivo_es_reciente"](str(ruta), dias_maximos=9) is False


# --------------------------------------------------------------------------
# _sbdb_cache_ok
# --------------------------------------------------------------------------

def _csv_sbdb(ruta, campos):
    pd.DataFrame([{c: 1 for c in campos}]).to_csv(ruta, index=False)


def test_sbdb_cache_ok_falso_si_no_existe(nb_datos, tmp_path):
    assert nb_datos["_sbdb_cache_ok"](str(tmp_path / "sbdb_neo.csv")) is False


def test_sbdb_cache_ok_verdadero_con_todos_los_campos(nb_datos, tmp_path):
    ruta = tmp_path / "sbdb_neo.csv"
    _csv_sbdb(ruta, nb_datos["sbdb_fields"])
    assert nb_datos["_sbdb_cache_ok"](str(ruta)) is True


def test_sbdb_cache_ok_falso_si_falta_un_campo(nb_datos, tmp_path):
    """El cache de una version anterior del notebook debe invalidarse."""
    ruta = tmp_path / "sbdb_neo.csv"
    _csv_sbdb(ruta, [c for c in nb_datos["sbdb_fields"] if c != "moid"])
    assert nb_datos["_sbdb_cache_ok"](str(ruta)) is False


def test_sbdb_cache_ok_falso_si_el_csv_esta_corrupto(nb_datos, tmp_path):
    ruta = tmp_path / "sbdb_neo.csv"
    ruta.write_bytes(b"\x00\x01\x02")
    assert nb_datos["_sbdb_cache_ok"](str(ruta)) is False


def test_sbdb_cache_ok_falso_si_esta_desactualizado(nb_datos, tmp_path):
    ruta = tmp_path / "sbdb_neo.csv"
    _csv_sbdb(ruta, nb_datos["sbdb_fields"])
    viejo = time.time() - 31 * 86400
    os.utime(ruta, (viejo, viejo))
    assert nb_datos["_sbdb_cache_ok"](str(ruta)) is False


# --------------------------------------------------------------------------
# descargar_cad
# --------------------------------------------------------------------------

class RespuestaFalsa:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def payload(filas):
    return {"count": len(filas), "fields": ["des", "cd", "dist"], "data": filas}


@pytest.fixture
def cad_falsa(monkeypatch):
    """Registra las peticiones y devuelve un evento por tramo solicitado."""
    llamadas = []

    def fake_get(url, params=None, timeout=None):
        llamadas.append({"url": url, "params": params, "timeout": timeout})
        anio = params["date-min"][:4]
        return RespuestaFalsa(payload([[f"obj-{anio}", f"{anio}-Jan-01 00:00", "0.1"]]))

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    return llamadas


def test_descargar_cad_trocea_el_rango_en_decadas(nb_datos, cad_falsa):
    df = nb_datos["descargar_cad"](dist_max=0.5, anio_ini=1900, paso=10)

    anio_actual = pd.Timestamp.utcnow().year
    esperados = list(range(1900, anio_actual + 1, 10))
    assert [c["params"]["date-min"] for c in cad_falsa] == \
        [f"{a}-01-01" for a in esperados]
    # ningun tramo pasa del anio siguiente al actual
    assert cad_falsa[-1]["params"]["date-max"] == f"{anio_actual + 1}-01-01"
    assert len(df) == len(esperados)
    assert list(df.columns) == ["des", "cd", "dist"]


def test_descargar_cad_propaga_los_parametros_de_la_consulta(nb_datos, cad_falsa):
    nb_datos["descargar_cad"](dist_max=0.5, anio_ini=2020, paso=1000, timeout=7)
    assert len(cad_falsa) == 1
    llamada = cad_falsa[0]
    assert llamada["url"] == "https://ssd-api.jpl.nasa.gov/cad.api"
    # dist-max explicito: dejarlo por defecto (0.05) censura la muestra
    assert llamada["params"]["dist-max"] == 0.5
    assert llamada["params"]["diameter"] == "true"
    assert llamada["timeout"] == 7


def test_descargar_cad_elimina_eventos_duplicados_en_la_frontera(nb_datos,
                                                                 monkeypatch):
    duplicado = ["433", "2010-Jan-01 00:00", "0.1"]
    monkeypatch.setattr(requests, "get", lambda *a, **k: RespuestaFalsa(
        payload([duplicado, list(duplicado), ["99942", "2011-Jan-01 00:00", "0.2"]])))
    df = nb_datos["descargar_cad"](dist_max=0.5, anio_ini=2020, paso=1000)
    assert len(df) == 2


def test_descargar_cad_ignora_tramos_vacios(nb_datos, monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: RespuestaFalsa({"count": 0}))
    df = nb_datos["descargar_cad"](dist_max=0.5, anio_ini=2020, paso=1000)
    assert df.empty


def test_descargar_cad_reintenta_tras_un_fallo_de_red(nb_datos, monkeypatch):
    esperas = []
    intentos = {"n": 0}

    def flaky_get(*a, **k):
        intentos["n"] += 1
        if intentos["n"] == 1:
            raise requests.exceptions.ConnectionError("corte de la respuesta")
        return RespuestaFalsa(payload([["433", "2010-Jan-01 00:00", "0.1"]]))

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setattr(time, "sleep", esperas.append)

    df = nb_datos["descargar_cad"](dist_max=0.5, anio_ini=2020, paso=1000)
    assert intentos["n"] == 2
    assert esperas == [2]  # espera creciente 2*(intento+1)
    assert len(df) == 1


def test_descargar_cad_propaga_el_error_tras_agotar_los_intentos(nb_datos,
                                                                monkeypatch):
    intentos = {"n": 0}

    def siempre_falla(*a, **k):
        intentos["n"] += 1
        raise requests.exceptions.Timeout("timeout")

    monkeypatch.setattr(requests, "get", siempre_falla)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    with pytest.raises(requests.exceptions.Timeout):
        nb_datos["descargar_cad"](dist_max=0.5, anio_ini=2020, paso=1000,
                                  intentos=3)
    assert intentos["n"] == 3
