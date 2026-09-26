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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neos import datos, etiquetas
from neos.constantes import (
    ALBEDO_ASUMIDO,
    DIR_DATOS,
    FACTOR_H_D,
    TOL_DESVIACION_MOID_PP,
    TOL_DIST_MAX,
    TOL_FRAC_CENSURA,
    UMBRAL_H,
    UMBRAL_MOID,
)

datos.configurar_salida_utf8()

# Los umbrales de estas comprobaciones salen de `neos.constantes` en vez de estar
# escritos aquí. Antes el script tenía sus propias copias, y fue así como una
# comprobación siguió midiendo con un criterio mientras el resto del proyecto ya
# usaba otro: dos números iguales en dos sitios que dejan de estarlo en cuanto
# alguien cambia uno de los dos.

fallos = []


def bloque(clave, texto):
    print(f"\n{'=' * 78}\n[{clave}] {texto}\n{'=' * 78}")


def veredicto(ok, mensaje):
    print(f"  {'✔ OK   ' if ok else '✘ FALLO'}  {mensaje}")
    if not ok:
        fallos.append(mensaje)


def info(mensaje):
    print(f"  · INFO   {mensaje}")


def cargar(ruta_cad=None, verificar=False):
    """Carga el catálogo y devuelve (df, observados, por_objeto).

    `verificar=False` por defecto: este script se ejecuta en tests con catálogos
    sintéticos en `tmp_path`, que no tienen nada que ver con el snapshot congelado y
    cuyo hash no está en `snapshot_info.json`. La verificación se hace una vez, en
    el entry point real (`main`), no en cada bloque.
    """
    try:
        df = datos.cargar_close_approaches(ruta=ruta_cad, solo_observadas=False,
                                           verificar=verificar)
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    obs = df[df["post_discovery"] == 1]
    obj = datos.agregar_por_objeto(
        obs, columnas=["distnom_min", "distmin_min", "H_obs", "H_sbdb", "moid",
                        "pha", "n_appro"], con_vinf_closest=False)
    return df, obs, obj


def a_censura(df, obj):
    """Las cinco señales de que el catálogo está censurado justo en el umbral PHA.

    Un catálogo descargado con el `dist-max` por defecto de la CAD API (0.05 au) no
    está sesgado: está *censurado*, y el sesgo se ve por cinco vías independientes.
    La más evidente es que el 100 % de los objetos queda dentro del umbral, lo que es
    imposible en una población real.
    """
    bloque("A", "CENSURA EN EL UMBRAL PHA  (corregida con dist-max=0.5)")
    dmax = df["CA DistanceNominal (au)"].max()
    veredicto(dmax > TOL_DIST_MAX,
              f"max(dist) = {dmax:.4f} au — la muestra llega más allá del umbral {UMBRAL_MOID}")

    frac = (obj.distnom_min <= UMBRAL_MOID).mean()
    veredicto(frac < TOL_FRAC_CENSURA,
              f"objetos con paso observado <= {UMBRAL_MOID} au: {100*frac:.1f}% "
              f"(censurado daba 100%)")

    v = obj.dropna(subset=["moid"])
    fm = (v.moid <= UMBRAL_MOID).mean()
    veredicto(fm < TOL_FRAC_CENSURA,
              f"objetos con MOID <= {UMBRAL_MOID} au: {100*fm:.1f}% (censurado daba 99.7%)")

    w = obj.dropna(subset=["pha", "H_sbdb"])
    acc = ((w.H_sbdb <= UMBRAL_H).astype(int) == w.pha).mean()
    veredicto(acc < TOL_FRAC_CENSURA,
              f"PHA == (H<={UMBRAL_H:g}) solo el {100*acc:.1f}% — la etiqueta ya no es un "
              f"umbral único (censurado daba 99.6%)")

    proxy = etiquetas.etiqueta_proxy(obj.H_obs, obj.distnom_min)
    igual = (proxy == (obj.H_obs <= UMBRAL_H).astype(int)).mean()
    veredicto(igual < TOL_FRAC_CENSURA,
              f"PHA_proxy == (H<={UMBRAL_H:g}) solo el {100*igual:.1f}% — la condición de "
              f"distancia ya discrimina (censurado daba 100%)")


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
    imput = np.isclose(d["Diameter(km)"], datos.diametro_desde_h(d["H(mag)"]), rtol=1e-6)
    info(f"Diameter sigue siendo imputado desde H en el {100*imput.mean():.1f}% de las filas")

    v = df.dropna(subset=["V relative(km/s)", "V infinity(km/s)", "CA DistanceNominal (au)"])
    obs = v["V relative(km/s)"] - v["V infinity(km/s)"]
    teo = (datos.v_relativa_teorica(v["V infinity(km/s)"], v["CA DistanceNominal (au)"])
           - v["V infinity(km/s)"])
    info(f"v_rel = f(v_inf, dist) con error mediano {(obs - teo).abs().median():.5f} km/s")
    info("Ambas se excluyen del conjunto exploratorio: quedan dist, v_inf y H.")


def d_sesgo_muestreo(obj, ruta_sbdb=None):
    """Compara la población del catálogo CAD con la de toda la SBDB.

    La comparación es por *condición*, no solo por prevalencia de PHA: la
    prevalencia es el producto de dos efectos (MOID y H) y se pueden cancelar, así
    que dos poblaciones con prevalencias iguales pueden tener sesgos opuestos.
    """
    bloque("D", "SESGO DE MUESTREO LEÍDO POR CONDICIÓN  (corregida)")
    ruta_sbdb = ruta_sbdb or os.path.join(DIR_DATOS, "sbdb_neo.csv")
    if not os.path.exists(ruta_sbdb):
        info("omitido: falta data/sbdb_neo.csv")
        return
    s = datos.marcar_en_catalogo(datos.cargar_sbdb(ruta_sbdb), obj.Object)

    print(f"  {'poblacion':<26} {'n':>8} {'PHA':>7} {'MOID':>11} {'H':>7}")
    ref = {}
    for lab, sub in [("todos los NEOs (SBDB)", s), ("en el catalogo CAD", s[s.en_cat])]:
        st = datos.estadisticas_poblacion(sub)
        ref[lab] = st
        print(f"  {lab:<26} {st['n']:>8,} {st['pha']:>6.1f}% "
              f"{st['moid']:>10.1f}% {st['h']:>6.1f}%")
    pobl, cat = ref["todos los NEOs (SBDB)"], ref["en el catalogo CAD"]
    for clave, etiqueta in (("moid", f"MOID<={UMBRAL_MOID}"), ("h", f"H<={UMBRAL_H:g}")):
        veredicto(abs(cat[clave] - pobl[clave]) < TOL_DESVIACION_MOID_PP,
                  f"desviacion en {etiqueta} respecto a la poblacion: "
                  f"{abs(cat[clave] - pobl[clave]):.1f} pp (censurado eran 45.6 pp)")


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
    """Cuánto de la etiqueta se puede reconstruir y cuánto no.

    El techo de exactitud importa para interpretar cualquier métrica: si la regla
    `MOID<=0.05 & H<=22` solo coincide con el flag oficial en el 99.87 %, entonces
    un clasificador que alcance esa cifra está reproduciendo el flag, no
    adivinando, y por encima de ella está midiendo algo que la propia definición
    no puede expresar.
    """
    bloque("I", "REPRODUCIBILIDAD DEL FLAG OFICIAL  (heredada: techo de exactitud)")
    w = obj.dropna(subset=["pha", "H_sbdb", "moid"])
    regla = etiquetas.etiqueta_pha(w.moid, w.H_sbdb)
    info(f"regla exacta (MOID<={UMBRAL_MOID} & H<={UMBRAL_H:g}) vs flag pha: "
         f"{100*(regla == w.pha).mean():.2f}% — techo de exactitud alcanzable")


def main(ruta_cad=None, ruta_sbdb=None, verificar=True):
    """Ejecuta los nueve bloques y devuelve 1 si algún bloque corregible falla.

    `verificar=True` comprueba el sha256 del snapshot antes de nada. Un script de
    integridad que no comprueba la integridad del fichero que está leyendo no tiene
    sentido: si el CSV cambia y el resto de comprobaciones siguen dando OK, eso mismo
    es un hallazgo.
    """
    if verificar:
        datos.verificar_snapshot(ruta_cad)
    df, obs, obj = cargar(ruta_cad=ruta_cad, verificar=False)
    print(f"Dataset: {len(df):,} eventos ({len(obs):,} observados), "
          f"{obj.Object.nunique():,} objetos con evento observado")
    a_censura(df, obj)
    b_h_definicional(df)
    c_features_redundantes(df)
    d_sesgo_muestreo(obj, ruta_sbdb=ruta_sbdb)
    e_dist_min(obj)
    f_retroactivas(df)
    g_moid_epoca(obj)
    h_albedo()
    i_flag(obj)

    print("\n" + "=" * 78)
    if fallos:
        print(f"{len(fallos)} COMPROBACION(ES) FALLIDA(S):")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    print("Todas las comprobaciones corregibles pasan.")
    print("Detalle e interpretacion en docs/05-discrepancias.md")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    main()
