"""Carga, verificación, descarga y agregación del catálogo de aproximaciones cercanas.

Tres cosas que antes vivían sueltas en tres sitios y ahora están aquí:

*   **Verificación del snapshot.** `data/snapshot_info.json` declara un sha256 que
    nadie comprobaba. `verificar_snapshot()` lo comprueba, de modo que "congelado"
    sea una propiedad verificada y no una afirmación.
*   **Agregación con corte temporal.** Las features se agregaban sobre *todos* los
    acercamientos post-descubrimiento, incluidos los posteriores a la fecha de
    decisión, y el holdout temporal por tanto no era temporal. `agregar_por_objeto()`
    acepta `hasta`/`desde` para cerrar eso.
*   **`vinf` del evento más cercano.** `vinf_max` es un estadístico de orden cuyo
    valor esperado crece con el número de muestras, así que codifica cuántas
    veces se observó el objeto en lugar de su cinemática. `vinf_closest` no tiene
    ese sesgo porque se evalúa sobre un único evento, el más cercano.
"""

import glob
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from neos.constantes import (
    AU_KM,
    DIST_MAX_AU,
    FACTOR_H_D,
    MU_TIERRA,
    RAIZ,
    RUTA_CAD,
    RUTA_SBDB,
    RUTA_SNAPSHOT_INFO,
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

# Columnas sin las que no se puede calcular nada. Se comprueban al cargar para
# fallar con un mensaje accionable en vez de con un KeyError a mitad del informe.
COLUMNAS_CAD_REQUERIDAS = (
    "Object",
    "Close-Approach (CA) Date",
    "CA DistanceNominal (au)",
    "CA DistanceMinimum (au)",
    "H(mag)",
    "H_SBDB(mag)",
    "MOID (au)",
    "PHA_official",
    "post_discovery",
    "V relative(km/s)",
    "V infinity(km/s)",
    "Diameter(km)",
)
COLUMNAS_SBDB_REQUERIDAS = ("pdes", "pha", "moid", "H")

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
    "moid": ("MOID (au)", "first"),
    "n_appro": ("Object", "size"),
    "first_obs_year": ("first_obs_year", "min"),
    "data_arc": ("data_arc(d)", "max"),
    "n_obs_used": ("n_obs_used", "max"),
    "pha": ("PHA_official", "max"),
}


# --------------------------------------------------------------------------- #
# Utilidades de bajo nivel
# --------------------------------------------------------------------------- #
def configurar_salida_utf8():
    """Pone la consola en UTF-8, de forma idempotente.

    La consola de Windows usa cp1252 y revienta con los símbolos de los informes
    (✔, ⚠, ±, σ). Antes cada script y cada notebook llamaba esto a mano en su
    primera celda, y bastaba con que uno lo olvidara para que el fallo apareciera
    a mitad de un informe, ya con resultados calculados. Se llama al importar el
    módulo para que el símbolo no dependa de que el llamante se acuerde.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # stream ya cerrado o sin soporte; no es motivo para fallar


configurar_salida_utf8()


def limpiar_fecha(fecha_str):
    """Elimina sufijos de incertidumbre como ±00:01 en las fechas de la API."""
    if isinstance(fecha_str, str):
        return fecha_str.split("±")[0].strip()
    return fecha_str


def _version_snapshot(ruta):
    """Fecha del snapshot extraída del nombre, para ordenar numéricamente.

    Ordenar por glob es orden lexicográfico: `v2026081` (7 dígitos) se ordenaría
    después de `v20260811` (8), y un sufijo como `_beta` rompe el patrón entero.
    """
    m = re.search(r"_v(\d{6,8})(?!\d)", os.path.basename(ruta))
    return int(m.group(1)) if m else -1


def _snapshots(ruta_base):
    """Rutas de los snapshots versionados de `ruta_base`, de más antiguo a más nuevo.

    Ordenar por glob es orden lexicográfico: `v2026081` (7 dígitos) se ordenaría
    después de `v20260811` (8), y un sufijo como `_beta` rompe el patrón entero. Por
    eso se extrae la fecha con un regex y se ordena por entero.
    """
    base = os.path.splitext(os.path.basename(ruta_base))[0]
    raiz = os.path.join(os.path.dirname(ruta_base), f"{base}_v*.csv")
    return sorted(glob.glob(raiz), key=_version_snapshot)


def resolver_ruta_cad(preferir_snapshot=True, ruta=None):
    """Ruta del CSV de aproximaciones cercanas que hay que analizar.

    `ruta` explícita gana siempre y no se toca si no existe: quien la pasa sabe qué
    fichero quiere, y un `FileNotFoundError` con esa ruta es el mensaje correcto.
    Antes, en cambio, una ruta explícita inexistente caía alSnapshot real, de modo
    que los tests con catálogos sintéticos acababan analysing los 340 469 eventos
    de producción en vez de los 5 que habían escrito.

    `preferir_snapshot=True` (el valor por defecto en todos los experimentos) hace
    que el snapshot congelado gane al CSV de trabajo. El CSV de trabajo se regenera
    desde la API en cuanto se ejecuta el notebook de datos, así que preferirlo
    llevaba a que los notebooks consumieran datos distintos de los del paper sin
    ningún aviso.
    """
    if ruta:
        return os.path.normpath(os.path.join(RAIZ, ruta)) \
            if not os.path.isabs(ruta) else ruta
    snaps = _snapshots(RUTA_CAD)
    candidatas = ([snaps[-1]] if snaps else []) + [RUTA_CAD]
    if not preferir_snapshot:
        candidatas = [RUTA_CAD] + ([snaps[-1]] if snaps else [])
    for c in candidatas:
        if os.path.exists(c):
            return c
    return None


def resolver_ruta_sbdb():
    snaps = _snapshots(RUTA_SBDB)
    for c in ([RUTA_SBDB] + ([snaps[-1]] if snaps else [])):
        if os.path.exists(c):
            return c
    return None


# --------------------------------------------------------------------------- #
# Snapshot: hash y metadatos
# --------------------------------------------------------------------------- #
def sha256_archivo(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def leer_snapshot_info(ruta=RUTA_SNAPSHOT_INFO):
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def verificar_snapshot(ruta_csv=None, verificar=True, estricto=True):
    """Comprueba que el CSV corresponde al sha256 declarado en snapshot_info.json.

    Devuelve un dict con el resultado. Con `estricto=True` aborta si no coincide,
    porque un snapshot que no verifica significa que todas las cifras derivadas
    son de otro dataset y no se pueden comparar con nada.
    """
    ruta = ruta_csv or resolver_ruta_cad()
    info = leer_snapshot_info()
    if ruta is None or info is None:
        return {"verificado": False, "motivo": "sin snapshot_info.json o sin CSV"}

    real = sha256_archivo(ruta)
    esperado = info.get("sha256")
    resultado = {
        "ruta": os.path.basename(ruta),
        "sha256_real": real,
        "sha256_esperado": esperado,
        "verificado": bool(esperado) and real == esperado,
        "snapshot_date_utc": info.get("snapshot_date_utc"),
        "n_events": info.get("n_events"),
        "n_objects": info.get("n_objects"),
        "dist_max_au": info.get("dist_max_au"),
    }
    if not verificar:
        return resultado
    if not resultado["verificado"]:
        mensaje = (
            f"El CSV {ruta} no corresponde a data/snapshot_info.json.\n"
            f"  esperado {esperado}\n  real     {real}\n"
            "El análisis publicado usa el snapshot congelado; regenera el dataset "
            "o vuelve al snapshot versionado."
        )
        if estricto:
            raise SystemExit(mensaje)
        print(f"⚠ AVISO  {mensaje}")
    else:
        print(f"✔ snapshot verificado: {resultado['ruta']} "
              f"({real[:16]}…, {info.get('n_events', 0):,} eventos)")
    return resultado


# --------------------------------------------------------------------------- #
# Físicas de referencia
# --------------------------------------------------------------------------- #
def diametro_desde_h(h):
    """Diámetro (km) imputado desde la magnitud absoluta con albedo asumido."""
    return FACTOR_H_D * 10 ** (-0.2 * h)


def v_relativa_teorica(v_inf, dist_au):
    """v_rel del encuentro hiperbólico: sqrt(v_inf² + 2·mu/r), con r en km."""
    return np.sqrt(v_inf ** 2 + 2 * MU_TIERRA / (dist_au * AU_KM))


def etiqueta_proxy(h, dist):
    """Proxy observacional. Se mantiene aquí por compatibilidad; la versión con
    nombre propio vive en `neos.etiquetas`."""
    return ((h <= UMBRAL_H) & (dist <= UMBRAL_MOID)).astype(int)


# --------------------------------------------------------------------------- #
# Caché y descarga
# --------------------------------------------------------------------------- #
def archivo_es_reciente(ruta, dias_maximos=30):
    """Devuelve True si el archivo fue modificado hace menos de dias_maximos días.

    El mtime es un criterio débil (un `git checkout` o un antivirus lo alteran), así
    que `cache_valido()` añade además una comprobación de esquema.
    """
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


class ErrorAPI(RuntimeError):
    """Fallo de la API que un `raise_for_status()` no detecta.

    La CAD y la SBDB responden 200 con un cuerpo JSON de error (`{"message": ...}`)
    cuando la consulta es demasiado grande o el parámetro no existe, y también
    devuelven JSON truncado si cortan la respuesta. Ambos casos pasan el
    `raise_for_status()` y romperían el dataset entero sin avisar.
    """


def payload_json(resp, contexto):
    """Extrae el JSON de una respuesta de la API validando que tenga contenido.

    Lanza `ErrorAPI` si viene un mensaje de error en un 200, o si el cuerpo no es
    JSON válido, o si `count` no coincide con el número de filas recibidas (truncado).
    """
    try:
        js = resp.json()
    except ValueError as exc:
        raise ErrorAPI(f"{contexto}: respuesta no es JSON ({exc})") from exc
    if isinstance(js, dict) and js.get("message") and "data" not in js:
        raise ErrorAPI(f"{contexto}: la API respondio con un error: {js['message']}")
    declarado, recibido = js.get("count"), len(js.get("data") or [])
    if declarado is not None and declarado != recibido:
        raise ErrorAPI(
            f"{contexto}: la API declara count={declarado} pero envio {recibido} "
            "filas. La respuesta esta truncada; hay que reintentar el tramo.")
    return js


def descargar_cad(dist_max=DIST_MAX_AU, anio_ini=1900, paso=10, intentos=3,
                  timeout=300, verbose=True):
    """Descarga aproximaciones cercanas de la CAD API por tramos de `paso` años.

    La consulta completa (~340k eventos) en una sola petición falla de forma
    intermitente por corte de la respuesta; trocearla la hace reproducible y, de
    paso, unas 3 veces más rápida. `date-min` es inclusivo y `date-max` exclusivo
    en la API, así que los tramos `[a, b)` son contiguos sin huecos ni solapes.
    """
    anio_fin = int(datetime.now(timezone.utc).strftime("%Y")) + 1
    tramos = [(a, min(a + paso, anio_fin)) for a in range(anio_ini, anio_fin, paso)]
    filas, campos, vacios = [], None, []

    for a, b in tramos:
        for intento in range(intentos):
            try:
                resp = requests.get(
                    URL_CAD,
                    params={"date-min": f"{a}-01-01", "date-max": f"{b}-01-01",
                            "dist-max": dist_max, "diameter": "true"},
                    timeout=timeout)
                resp.raise_for_status()
                js = payload_json(resp, f"CAD {a}-{b}")
                if js.get("count"):
                    campos = js["fields"]
                    filas.extend(js["data"])
                else:
                    vacios.append((a, b))
                break
            except (requests.exceptions.RequestException, ErrorAPI) as exc:
                if intento == intentos - 1:
                    raise
                if verbose:
                    print(f"  reintento {intento + 1}/{intentos} en {a}-{b}: {exc}")
                time.sleep(2 * (intento + 1))
        if verbose:
            print(f"  {a}-{b}: {len(filas):,} eventos acumulados")

    if not filas:
        raise SystemExit("La CAD API no devolvió ningún evento. Revisa dist-max y el rango de fechas.")
    if vacios and verbose:
        print(f"  tramos sin eventos (normal en fechas sin catálogo): {vacios}")
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


def descargar_sbdb(ruta=RUTA_SBDB, dias_maximos=30, intentos=3, timeout=180):
    """Catálogo de NEOs de la SBDB, desde cache local reciente o desde la API.

    A diferencia de la CAD, esto va en una sola petición de ~43k filas, así que es
    el punto único de fallo duro del pipeline: por eso tiene reintentos y un
    timeout holgado (la petición tarda ~45 s).
    """
    if cache_valido(ruta, CAMPOS_SBDB, dias_maximos=dias_maximos):
        print(f"✔ Catalogo SBDB local reciente. Cargando {ruta}...")
        return pd.read_csv(ruta)

    print("⬇ Descargando catalogo de NEOs desde la SBDB de JPL...")
    for intento in range(intentos):
        try:
            resp = requests.get(URL_SBDB,
                                params={"fields": ",".join(CAMPOS_SBDB), "sb-group": "neo"},
                                timeout=timeout)
            resp.raise_for_status()
            js = payload_json(resp, "SBDB")
            sbdb = pd.DataFrame(js["data"], columns=js["fields"])
            sbdb.to_csv(ruta, index=False)
            print(f"✔ Catalogo SBDB guardado. {len(sbdb):,} NEOs.")
            return sbdb
        except (requests.exceptions.RequestException, ErrorAPI) as exc:
            if intento == intentos - 1:
                raise
            print(f"  reintento {intento + 1}/{intentos}: {exc}")
            time.sleep(2 * (intento + 1))


def normalizar_sbdb(sbdb):
    """Tipos utilizables: designación como str, numéricos, pha 0/1 y año de 1ª obs.

    `first_obs` se lee solo si está presente. Un CSV de la SBDB con el subconjunto de
    campos que pide el bloque D no lo trae, y exigirlo hacía que esa única
    comprobación fuera la que fallaba en los tests, por un `KeyError` en una columna
    que no interviene en nada de lo que el bloque mide.
    """
    sbdb = sbdb.copy()
    sbdb["pdes"] = sbdb["pdes"].astype(str)
    for col in ("moid", "H", "data_arc", "n_obs_used"):
        if col in sbdb.columns:
            sbdb[col] = pd.to_numeric(sbdb[col], errors="coerce")
    if "pha" in sbdb.columns:
        sbdb["pha01"] = sbdb["pha"].map({"Y": 1, "N": 0})
    if "first_obs" in sbdb.columns:
        sbdb["first_obs_year"] = pd.to_datetime(sbdb["first_obs"], errors="coerce").dt.year
    return sbdb


def cargar_sbdb(ruta=None):
    """Lee y normaliza el CSV local de la SBDB (sin tocar la red)."""
    ruta = ruta or resolver_ruta_sbdb()
    if not ruta:
        raise FileNotFoundError(
            f"No se encontró el catálogo de la SBDB en {RUTA_SBDB}. "
            "Ejecuta la celda de la SBDB en data/ProyectoNeoRework_data.ipynb")
    return normalizar_sbdb(pd.read_csv(ruta))


def marcar_en_catalogo(sbdb, objetos):
    """Añade `en_cat`: si la designación aparece en el catálogo CAD."""
    sbdb = sbdb.copy()
    sbdb["en_cat"] = sbdb["pdes"].astype(str).isin(set(pd.Series(objetos).astype(str)))
    return sbdb


def estadisticas_poblacion(sub):
    """Prevalencia PHA y cada condición de la definición, en %, para una población.

    El sesgo hay que leerlo en CADA condición por separado: la prevalencia PHA es su
    producto y los dos efectos pueden cancelarse, así que dos poblaciones con igual
    prevalencia pueden tener sesgos opuestos en MOID y en H.

    Cada fracción se calcula sobre los objetos con ese dato, no sobre toda la
    población. La SBDB deja `moid` y `H` vacíos en los objetos sin solución orbital,
    y usar el denominador completo bajaría las tres cifras en la misma proporción,
    con lo que la comparación deja de informar.
    """
    con_moid = sub.dropna(subset=["moid"])
    con_h = sub.dropna(subset=["H"])
    con_pha = sub.dropna(subset=["pha01"])
    return {
        "n": len(sub),
        "pha": 100 * con_pha["pha01"].mean() if len(con_pha) else float("nan"),
        "moid": (100 * (con_moid["moid"] <= UMBRAL_MOID).mean()
                 if len(con_moid) else float("nan")),
        "h": 100 * (con_h["H"] <= UMBRAL_H).mean() if len(con_h) else float("nan"),
        "n_con_moid": len(con_moid),
        "n_con_h": len(con_h),
        "n_con_pha": len(con_pha),
    }


# --------------------------------------------------------------------------- #
# Carga y agregación
# --------------------------------------------------------------------------- #
def leer_csv(ruta, requeridas=(), como_regenerar=""):
    """Lee un CSV del dataset validando que sea utilizable.

    Un CSV truncado, vacío o de una versión anterior del pipeline produciría aquí
    un KeyError o una división por cero varios bloques más adelante, con un
    mensaje que no dice que hay que regenerarlo.
    """
    if not os.path.exists(ruta):
        raise SystemExit(f"Falta {ruta}. {como_regenerar}")
    try:
        df = pd.read_csv(ruta)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise SystemExit(
            f"No se pudo leer {ruta} ({type(exc).__name__}: {exc}). {como_regenerar}")
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise SystemExit(f"{ruta} no tiene las columnas {faltan}. {como_regenerar}")
    if df.empty:
        raise SystemExit(f"{ruta} no tiene filas. {como_regenerar}")
    return df


def cargar_close_approaches(ruta=None, solo_observadas=True, preferir_snapshot=True,
                            verificar=True):
    """Lee el CSV de aproximaciones cercanas, con verificación de snapshot.

    Por defecto restringe a los eventos posteriores al inicio del arco observado
    del objeto: el resto son integraciones numéricas hacia atrás. Nótese que
    "posterior al arco" no es "observado" en el sentido de medido por un sensor:
    `dist`, `v_rel` y `v_inf` los calcula JPL integrando la solución orbital.
    """
    ruta = resolver_ruta_cad(preferir_snapshot=preferir_snapshot, ruta=ruta)
    if ruta is None:
        raise SystemExit(
            f"No se encontró el CSV en {RUTA_CAD} ni ningún snapshot en data/. "
            "Ejecuta primero data/ProyectoNeoRework_data.ipynb")
    if not os.path.exists(ruta):
        raise SystemExit(
            f"No existe el CSV en {ruta}. "
            "Ejecuta primero data/ProyectoNeoRework_data.ipynb, o pasa la ruta "
            "de un snapshot versionado.")
    if verificar:
        verificar_snapshot(ruta)

    df = pd.read_csv(ruta)
    if "post_discovery" not in df.columns:
        raise SystemExit("El CSV no tiene 'post_discovery': regenera con el notebook corregido.")
    df["Close-Approach (CA) Date"] = df["Close-Approach (CA) Date"].apply(limpiar_fecha)
    df["ca_date"] = pd.to_datetime(df["Close-Approach (CA) Date"], errors="coerce")
    df["ca_year"] = df["ca_date"].dt.year
    return df[df["post_discovery"] == 1].copy() if solo_observadas else df


def agregar_por_objeto(df, columnas=None, exigir_pha=False, hasta=None, desde=None,
                       con_vinf_closest=True):
    """Un registro por asteroide con los estadísticos de `AGREGACION_OBJETO`.

    `hasta` / `desde` cortan los eventos por la fecha del propio acercamiento, no
    por la del descubrimiento del objeto. Sin ese corte, un objeto descubierto en
    2005 entra en el conjunto de entrenamiento con sus acercamientos de 2020, y el
    holdout temporal deja de ser temporal.

    `con_vinf_closest` añade `vinf_closest`: la v∞ del evento con menor distancia
    nominal. Es la alternativa a `vinf_max`, cuyo valor esperado crece con el
    número de muestras (corr(vinf_max, n_appro) = +0.31 frente a +0.02 para la
    mediana), de modo que `vinf_max` mide cuántas veces se vio el objeto.
    """
    columnas = list(AGREGACION_OBJETO) if columnas is None else list(columnas)
    df = df.copy()
    if "dist_unc" not in df.columns:
        df["dist_unc"] = (df["CA DistanceNominal (au)"]
                          - df["CA DistanceMinimum (au)"])
    if "ca_date" not in df.columns and "Close-Approach (CA) Date" in df.columns:
        df["ca_date"] = pd.to_datetime(df["Close-Approach (CA) Date"], errors="coerce")
    if hasta is not None:
        df = df[df["ca_date"].dt.year < hasta]
    if desde is not None:
        df = df[df["ca_date"].dt.year >= desde]

    if con_vinf_closest:
        df_ord = df.sort_values(["Object", "CA DistanceNominal (au)", "ca_date"])
        df_ord = df_ord.drop_duplicates("Object", keep="first").set_index("Object")
        if "vinf_closest" not in columnas:
            columnas = columnas + ["vinf_closest"]

    spec = {}
    for nombre in columnas:
        if nombre == "vinf_closest":
            continue
        col, agg = AGREGACION_OBJETO[nombre]
        if col in df.columns:
            spec[nombre] = (col, agg)

    obj = df.groupby("Object").agg(**spec).reset_index()
    if con_vinf_closest:
        obj["vinf_closest"] = obj["Object"].map(df_ord["V infinity(km/s)"])
    if exigir_pha:
        obj = obj.dropna(subset=["pha"])
        obj["pha"] = obj["pha"].astype(int)
    return obj


def construir_objetos(ruta=None, hasta=None, desde=None, exigir_pha=False,
                      con_vinf_closest=True, verificar=True):
    """Atajo de `cargar_close_approaches` + `agregar_por_objeto` para uso habitual."""
    df = cargar_close_approaches(ruta=ruta, verificar=verificar)
    return agregar_por_objeto(df, hasta=hasta, desde=desde, exigir_pha=exigir_pha,
                              con_vinf_closest=con_vinf_closest)
