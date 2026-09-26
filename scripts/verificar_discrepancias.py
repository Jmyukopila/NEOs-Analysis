"""Control de integridad del dataset frente a las discrepancias documentadas.

Cada bloque comprueba una de las discrepancias de docs/05-discrepancias.md y
reporta OK / FALLO / INFO. Las discrepancias corregidas deben dar OK; las
heredadas (de la definición oficial de PHA o de los datos de JPL) se reportan
como INFO con su magnitud, para poder citarlas como limitaciones.

Uso (desde la raíz del repo, tras ejecutar el notebook de datos):

    python scripts/verificar_discrepancias.py
"""

import math
import os
import sys

import numpy as np
import pandas as pd

import glob

# La consola de Windows usa cp1252 por defecto y no admite los símbolos del informe
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(RAIZ, "data")

def resolver_ruta_cad():
    ruta_std = os.path.join(DIR_DATA, "close_approaches.csv")
    if os.path.exists(ruta_std):
        return ruta_std
    snaps = sorted(glob.glob(os.path.join(DIR_DATA, "close_approaches_v*.csv")))
    if snaps:
        return snaps[-1]
    return None

def resolver_ruta_sbdb():
    ruta_std = os.path.join(DIR_DATA, "sbdb_neo.csv")
    if os.path.exists(ruta_std):
        return ruta_std
    snaps = sorted(glob.glob(os.path.join(DIR_DATA, "sbdb_neo_v*.csv")))
    if snaps:
        return snaps[-1]
    return None

ALBEDO_ASUMIDO = 0.14
FACTOR_H_D = 1329 / math.sqrt(ALBEDO_ASUMIDO)  # ≈ 3552
UMBRAL_MOID = 0.05  # au
UMBRAL_H = 22.0  # mag

fallos = []


def bloque(clave, texto):
    print(f"\n{'=' * 78}\n[{clave}] {texto}\n{'=' * 78}")


def veredicto(ok, mensaje):
    print(f"  {'✔ OK   ' if ok else '✘ FALLO'}  {mensaje}")
    if not ok:
        fallos.append(mensaje)


def info(mensaje):
    print(f"  · INFO   {mensaje}")


def cargar():
    ruta_cad = resolver_ruta_cad()
    if not ruta_cad:
        sys.exit("Falta close_approaches.csv o su snapshot. Ejecuta primero data/ProyectoNeoRework_data.ipynb")
    df = pd.read_csv(ruta_cad)
    if "post_discovery" not in df.columns:
        sys.exit("El CSV no tiene 'post_discovery': regenera con el notebook corregido.")
    obs = df[df["post_discovery"] == 1]
    obj = obs.groupby("Object").agg(
        distnom_min=("CA DistanceNominal (au)", "min"),
        distmin_min=("CA DistanceMinimum (au)", "min"),
        H_obs=("H(mag)", "min"),
        H_sbdb=("H_SBDB(mag)", "max"),
        moid=("MOID (au)", "max"),
        pha=("PHA_official", "max"),
        n_appro=("Object", "size"),
    ).reset_index()
    return df, obs, obj


def a_censura(df, obj):
    bloque("A", "CENSURA EN EL UMBRAL PHA  (corregida con dist-max=0.5)")
    dmax = df["CA DistanceNominal (au)"].max()
    veredicto(dmax > 0.06,
              f"max(dist) = {dmax:.4f} au — la muestra llega más allá del umbral 0.05")

    frac = (obj.distnom_min <= UMBRAL_MOID).mean()
    veredicto(frac < 0.95,
              f"objetos con paso observado <= 0.05 au: {100*frac:.1f}% (censurado daba 100%)")

    v = obj.dropna(subset=["moid"])
    fm = (v.moid <= UMBRAL_MOID).mean()
    veredicto(fm < 0.95,
              f"objetos con MOID <= 0.05 au: {100*fm:.1f}% (censurado daba 99.7%)")

    w = obj.dropna(subset=["pha", "H_sbdb"])
    acc = ((w.H_sbdb <= UMBRAL_H).astype(int) == w.pha).mean()
    veredicto(acc < 0.95,
              f"PHA == (H<=22) solo el {100*acc:.1f}% — la etiqueta ya no es un umbral único "
              f"(censurado daba 99.6%)")

    proxy = ((obj.H_obs <= UMBRAL_H) & (obj.distmin_min <= UMBRAL_MOID)).astype(int)
    igual = (proxy == (obj.H_obs <= UMBRAL_H).astype(int)).mean()
    veredicto(igual < 0.95,
              f"PHA_proxy == (H<=22) solo el {100*igual:.1f}% — la condición de distancia "
              f"ya discrimina (censurado daba 100%)")


def b_h_definicional(df):
    bloque("B", "H ES UNA DE LAS DOS VARIABLES QUE DEFINEN PHA  (heredada, declarada)")
    d = df.dropna(subset=["H(mag)", "H_SBDB(mag)"])
    info(f"corr(H_CAD, H_SBDB) = {d['H(mag)'].corr(d['H_SBDB(mag)']):.6f}; "
         f"max|dif| = {(d['H(mag)'] - d['H_SBDB(mag)']).abs().max():.4f} mag")
    info("H no es una observación independiente: es el H de catálogo. Por eso el")
    info("trabajo se plantea como SUSTITUCIÓN DE MEDIDA y no como predicción.")


def c_features_redundantes(df):
    bloque("C", "FEATURES DERIVADAS FUERA DEL BLOQUE EXPLORATORIO  (corregida)")
    d = df.dropna(subset=["Diameter(km)", "H(mag)"])
    imput = np.isclose(d["Diameter(km)"], FACTOR_H_D * 10 ** (-0.2 * d["H(mag)"]), rtol=1e-6)
    info(f"Diameter sigue siendo imputado desde H en el {100*imput.mean():.1f}% de las filas")

    v = df.dropna(subset=["V relative(km/s)", "V infinity(km/s)", "CA DistanceNominal (au)"])
    obs = v["V relative(km/s)"] - v["V infinity(km/s)"]
    r = v["CA DistanceNominal (au)"] * 1.495978707e8
    teo = np.sqrt(v["V infinity(km/s)"] ** 2 + 2 * 3.986004418e5 / r) - v["V infinity(km/s)"]
    info(f"v_rel = f(v_inf, dist) con error mediano {(obs - teo).abs().median():.5f} km/s")
    info("Ambas se excluyen del conjunto exploratorio: quedan dist, v_inf y H.")


def d_sesgo_muestreo(obj):
    bloque("D", "SESGO DE MUESTREO LEÍDO POR CONDICIÓN  (corregida)")
    ruta_sbdb = resolver_ruta_sbdb()
    if not ruta_sbdb:
        info("omitido: falta data/sbdb_neo.csv o su snapshot")
        return
    s = pd.read_csv(ruta_sbdb)
    for c in ("moid", "H"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["pha01"] = s["pha"].map({"Y": 1, "N": 0})
    s["en_cat"] = s["pdes"].astype(str).isin(set(obj.Object.astype(str)))

    print(f"  {'población':<26} {'n':>8} {'PHA':>7} {'MOID<=0.05':>11} {'H<=22':>7}")
    ref = {}
    for lab, sub in [("todos los NEOs (SBDB)", s), ("en el catálogo CAD", s[s.en_cat])]:
        v = sub.dropna(subset=["moid"])
        ref[lab] = (100 * (v.moid <= UMBRAL_MOID).mean(), 100 * (sub.H <= UMBRAL_H).mean())
        print(f"  {lab:<26} {len(sub):>8,} {100*sub.pha01.mean():>6.1f}% "
              f"{ref[lab][0]:>10.1f}% {ref[lab][1]:>6.1f}%")
    dm = abs(ref["en el catálogo CAD"][0] - ref["todos los NEOs (SBDB)"][0])
    veredicto(dm < 25, f"desviación en MOID<=0.05 respecto a la población: {dm:.1f} pp "
                       f"(censurado eran 45.6 pp)")


def e_dist_min(obj):
    bloque("E", "dist_min ES UNA COTA 3-SIGMA  (corregida: se usa la nominal)")
    info(f"objetos con min(dist_min) < 1e-5 au: {(obj.distmin_min < 1e-5).sum()} "
         f"(siguen siendo enunciados sobre la incertidumbre, no pasos rasantes)")
    info("El conjunto de features usa distnom_min; la separación nominal-3σ entra")
    info("aparte como dist_unc_med, variable de calidad orbital.")


def f_retroactivas(df):
    bloque("F", "APROXIMACIONES CALCULADAS vs OBSERVADAS  (corregida: flag + filtro)")
    n_obs = int((df["post_discovery"] == 1).sum())
    veredicto("post_discovery" in df.columns,
              f"eventos observados: {n_obs:,} de {len(df):,} ({100*n_obs/len(df):.1f}%)")
    info("El análisis principal usa solo los observados; el catálogo completo queda")
    info("para el anexo de sensibilidad.")


def g_moid_epoca(obj):
    bloque("G", "EL MOID DEPENDE DE ÉPOCA  (heredada, no corregible)")
    v = obj.dropna(subset=["moid", "distnom_min"])
    frac = (v.moid > v.distnom_min + 1e-9).mean()
    info(f"objetos con MOID > distancia observada mínima: {100*frac:.1f}% — "
         f"imposible a época fija")
    info("Evidencia de evolución secular del MOID sobre la ventana 1900-2026.")


def h_albedo():
    bloque("H", "ALBEDO ÚNICO 0.14 FRENTE A DISTRIBUCIÓN BIMODAL  (heredada)")
    for p, lab in [(0.030, "población oscura (25.3%)"), (0.168, "población clara (74.7%)")]:
        info(f"p_V={p:<6} {lab:<26} D_real/D_asumido = {math.sqrt(ALBEDO_ASUMIDO/p):.2f}x")
    info("Limitación de la definición oficial de PHA, que es albedo-ciega.")


def i_flag(obj):
    bloque("I", "REPRODUCIBILIDAD DEL FLAG OFICIAL  (heredada: techo de exactitud)")
    w = obj.dropna(subset=["pha", "H_sbdb", "moid"])
    regla = ((w.H_sbdb <= UMBRAL_H) & (w.moid <= UMBRAL_MOID)).astype(int)
    info(f"regla exacta (H<=22 & MOID<=0.05) vs flag pha: {100*(regla == w.pha).mean():.2f}% "
         f"— techo de exactitud alcanzable")


def main():
    df, obs, obj = cargar()
    print(f"Dataset: {len(df):,} eventos ({len(obs):,} observados), "
          f"{obj.Object.nunique():,} objetos con evento observado")
    a_censura(df, obj)
    b_h_definicional(df)
    c_features_redundantes(df)
    d_sesgo_muestreo(obj)
    e_dist_min(obj)
    f_retroactivas(df)
    g_moid_epoca(obj)
    h_albedo()
    i_flag(obj)

    print("\n" + "=" * 78)
    if fallos:
        print(f"{len(fallos)} COMPROBACIÓN(ES) FALLIDA(S):")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    print("Todas las comprobaciones corregibles pasan.")
    print("Detalle e interpretación en docs/05-discrepancias.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
