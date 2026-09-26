"""Carga, descarga y agregación del catálogo de aproximaciones cercanas."""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from neos.constantes import (
    AU_KM,
    DIST_MAX_AU,
    FACTOR_H_D,
    MU_TIERRA,
    RUTA_CAD,
    RUTA_SBDB,
    UMBRAL_H,
    UMBRAL_MOID,
)

URL_CAD = "https://ssd-api.jpl.nasa.gov/cad.api"
URL_SBDB = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

CAMPOS_SBDB = ["pdes", "full_name", "pha", "neo", "moid", "H", "diameter",
               "first_obs", "last_obs", "data_arc", "n_obs_used"]

# Nombres de la CAD API -> nombres del CSV del proyecto
RENOMBRE_CAD = {
    "des": "Object",
    "cd": "Close-Approach (CA) Date",
    "dist": "CA DistanceNominal (au)",
    "dist_min": "CA DistanceMinimum (au)",
    "v_rel": "V relative(km/s)",
    "v_inf": "V infinity(km/s)",
    "h": "H(mag)",
    "diameter": "Diameter(km)",
    "diameter_sigma": "Std Diameter(km)",
}
COLUMNAS_CAD = list(RENOMBRE_CAD)

# Agregación a nivel objeto: la peligrosidad es propiedad del objeto, no del
# evento. Cada consumidor elige el subconjunto de columnas que necesita.
AGREGACION_OBJETO = {
    "distnom_min": ("CA DistanceNominal (au)", "min"),
    "distmin_min": ("CA DistanceMinimum (au)", "min"),
    "dist_unc_med": ("dist_unc", "median"),
    "vrel_max": ("V relative(km/s)", "max"),
    "vinf_med": ("V infinity(km/s)", "median"),
    "vinf_max": ("V infinity(km/s)", "max"),
    "H_obs": ("H(mag)", "min"),
    "H_sbdb": ("H_SBDB(mag)", "max"),
    "diam_max": ("Diameter(km)", "max"),
    "moid": ("MOID (au)", "max"),
    "n_appro": ("Object", "size"),
    "first_obs_year": ("first_obs_year", "max"),
    "data_arc": ("data_arc(d)", "max"),
    "pha": ("PHA_official", "max"),
}


def configurar_salida_utf8():
    """La consola de Windows usa cp1252 y no admite los símbolos de los informes."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def limpiar_fecha(fecha_str):
    """Elimina sufijos de incertidumbre como ±00:01 en las fechas de la API."""
    if isinstance(fecha_str, str):
        return fecha_str.split("±")[0].strip()
    return fecha_str


def archivo_es_reciente(ruta, dias_maximos=30):
    """Devuelve True si el archivo fue modificado hace menos de dias_maximos días."""
    if not os.path.exists(ruta):
        return False
    return time.time() - os.path.getmtime(ruta) < dias_maximos * 86400


def cache_valido(ruta, campos=(), dias_maximos=30):
    """Cache utilizable: reciente, legible y con todas las columnas pedidas."""
    if not archivo_es_reciente(ruta, dias_maximos=dias_maximos):
        return False
    try:
        return set(campos).issubset(pd.read_csv(ruta, nrows=1).columns)
    except Exception:
        return False


def diametro_desde_h(h):
    """Diámetro (km) imputado desde la magnitud absoluta con albedo asumido."""
    return FACTOR_H_D * 10 ** (-0.2 * h)


def v_relativa_teorica(v_inf, dist_au):
    """v_rel del encuentro hiperbólico: sqrt(v_inf² + 2·mu/r), con r en km."""
    return np.sqrt(v_inf ** 2 + 2 * MU_TIERRA / (dist_au * AU_KM))


def etiqueta_proxy(h, dist):
    """Etiqueta PHA derivada solo de lo observado: H <= 22 y distancia <= 0.05 au."""
    return ((h <= UMBRAL_H) & (dist <= UMBRAL_MOID)).astype(int)


def descargar_cad(dist_max=DIST_MAX_AU, anio_ini=1900, paso=10, intentos=3, timeout=300):
    """Descarga aproximaciones cercanas de la CAD API por tramos de `paso` años.

    La consulta completa (~340k eventos) en una sola petición falla de forma
    intermitente por corte de la respuesta; trocearla la hace reproducible y,
    de paso, unas 3 veces más rápida.
    """
    anio_fin = int(datetime.utcnow().strftime("%Y")) + 1
    tramos = [(a, min(a + paso, anio_fin)) for a in range(anio_ini, anio_fin, paso)]
    filas, campos = [], None

    for a, b in tramos:
        for intento in range(intentos):
            try:
                resp = requests.get(
                    URL_CAD,
                    params={"date-min": f"{a}-01-01", "date-max": f"{b}-01-01",
                            "dist-max": dist_max, "diameter": "true"},
                    timeout=timeout)
                resp.raise_for_status()
                js = resp.json()
                if js.get("count"):
                    campos = js["fields"]
                    filas.extend(js["data"])
                break
            except requests.exceptions.RequestException:
                if intento == intentos - 1:
                    raise
                time.sleep(2 * (intento + 1))
        print(f"  {a}-{b}: {len(filas):,} eventos acumulados", end="\r")

    print(" " * 60, end="\r")
    # Red de seguridad por si un evento cae justo en la frontera de dos tramos
    return pd.DataFrame(filas, columns=campos).drop_duplicates(subset=["des", "cd"])


def preparar_cad(df):
    """Selecciona columnas, convierte a numérico, imputa diámetro y renombra."""
    columnas = [c for c in COLUMNAS_CAD if c in df.columns]
    df = df[columnas].copy()

    for col in columnas:
        if col not in ("des", "cd"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sin_diametro = df["diameter"].isna()
    df.loc[sin_diametro, "diameter"] = diametro_desde_h(df.loc[sin_diametro, "h"])
    df.loc[df["diameter_sigma"].isna(), "diameter_sigma"] = df["diameter"] * 0.35

    return df.rename(columns=RENOMBRE_CAD)


def descargar_sbdb(ruta=RUTA_SBDB, dias_maximos=30, timeout=120):
    """Catálogo de NEOs de la SBDB, desde cache local reciente o desde la API."""
    if cache_valido(ruta, CAMPOS_SBDB, dias_maximos=dias_maximos):
        print(f"✔ Catalogo SBDB local reciente. Cargando {ruta}...")
        return pd.read_csv(ruta)

    print("⬇ Descargando catalogo de NEOs desde la SBDB de JPL...")
    resp = requests.get(URL_SBDB,
                        params={"fields": ",".join(CAMPOS_SBDB), "sb-group": "neo"},
                        timeout=timeout)
    resp.raise_for_status()
    js = resp.json()
    sbdb = pd.DataFrame(js["data"], columns=js["fields"])
    sbdb.to_csv(ruta, index=False)
    print(f"✔ Catalogo SBDB guardado. {len(sbdb):,} NEOs.")
    return sbdb


def normalizar_sbdb(sbdb):
    """Tipos utilizables: designación como str, numéricos, pha 0/1 y año de 1ª obs."""
    sbdb = sbdb.copy()
    sbdb["pdes"] = sbdb["pdes"].astype(str)
    for col in ("moid", "H", "data_arc", "n_obs_used"):
        if col in sbdb.columns:
            sbdb[col] = pd.to_numeric(sbdb[col], errors="coerce")
    sbdb["pha01"] = sbdb["pha"].map({"Y": 1, "N": 0})
    sbdb["first_obs_year"] = pd.to_datetime(sbdb["first_obs"], errors="coerce").dt.year
    return sbdb


def cargar_sbdb(ruta=RUTA_SBDB):
    """Lee y normaliza el CSV local de la SBDB (sin tocar la red)."""
    return normalizar_sbdb(pd.read_csv(ruta))


def marcar_en_catalogo(sbdb, objetos):
    """Añade `en_cat`: si la designación aparece en el catálogo CAD."""
    sbdb = sbdb.copy()
    sbdb["en_cat"] = sbdb["pdes"].astype(str).isin(set(pd.Series(objetos).astype(str)))
    return sbdb


def estadisticas_poblacion(sub):
    """Prevalencia PHA y cada condición de la definición, en %, para una población.

    El sesgo hay que leerlo en CADA condición por separado: la prevalencia PHA es
    su producto y los dos efectos pueden cancelarse.
    """
    con_moid = sub.dropna(subset=["moid"])
    return {
        "n": len(sub),
        "pha": 100 * sub["pha01"].mean(),
        "moid": 100 * (con_moid["moid"] <= UMBRAL_MOID).mean(),
        "h": 100 * (sub["H"] <= UMBRAL_H).mean(),
    }


def cargar_close_approaches(ruta=RUTA_CAD, solo_observadas=False):
    """Lee el CSV de aproximaciones cercanas que produce el notebook de datos.

    Normaliza la fecha y, con `solo_observadas`, restringe a los eventos
    posteriores al descubrimiento del objeto (el resto son integraciones
    numéricas hacia atrás, no observaciones).
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el CSV en {ruta}. "
            f"Ejecutar primero data/ProyectoNeoRework_data.ipynb")

    df = pd.read_csv(ruta)
    if "post_discovery" not in df.columns:
        raise ValueError(
            "El CSV no tiene 'post_discovery': regenera con el notebook corregido.")

    df["Close-Approach (CA) Date"] = df["Close-Approach (CA) Date"].apply(limpiar_fecha)
    return df[df["post_discovery"] == 1].copy() if solo_observadas else df


def agregar_por_objeto(df, columnas=None, exigir_pha=False):
    """Un registro por asteroide con los estadísticos de `AGREGACION_OBJETO`.

    `dist_unc_med` (ancho del intervalo 3σ: nominal − mínima, que mide
    incertidumbre orbital y no proximidad) se calcula aquí si se pide.
    """
    columnas = list(AGREGACION_OBJETO) if columnas is None else list(columnas)
    if "dist_unc_med" in columnas and "dist_unc" not in df.columns:
        df = df.assign(dist_unc=df["CA DistanceNominal (au)"]
                       - df["CA DistanceMinimum (au)"])

    spec = {}
    for nombre in columnas:
        col, agg = AGREGACION_OBJETO[nombre]
        if col in df.columns:
            spec[nombre] = (col, agg)

    obj = df.groupby("Object").agg(**spec).reset_index()
    if exigir_pha:
        obj = obj.dropna(subset=["pha"])
        obj["pha"] = obj["pha"].astype(int)
    return obj
