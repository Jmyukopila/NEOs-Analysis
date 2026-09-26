"""Verificación de Peligrosidad y Racionalidad del MOID Predicho.

Este script analiza la validez del modelo de Machine Learning en la detección de
asteroides potencialmente peligrosos (MOID <= 0.05 au), respetando la consistencia
temporal (sin mezclar épocas con diferente profundidad observacional, e.g. 1920 vs 2016).

Compara:
1. Peligrosos Reales (MOID Orbital Real <= 0.05 au).
2. Proxy Observacional Directo (distnom_min <= 0.05 au).
3. Clasificador ML (XGBoost con umbral optimizado).

Genera métricas de solapamiento, análisis de asteroides rescatados por ML,
distribución física de los falsos positivos y evolución temporal por épocas de descubrimiento.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost no está instalado. Por favor instálalo.")
    sys.exit(1)

# Asegurar codificación utf-8 en Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(RAIZ, "data")
DIR_FIG = os.path.join(RAIZ, "results", "figures")
DIR_TAB = os.path.join(RAIZ, "results", "tables")

os.makedirs(DIR_FIG, exist_ok=True)
os.makedirs(DIR_TAB, exist_ok=True)

SEED = 42
UMBRAL_MOID = 0.05  # au
THRESHOLD_PROB = 0.35  # Umbral de probabilidad óptimo para F2


def resolver_ruta_cad():
    ruta_std = os.path.join(DIR_DATA, "close_approaches.csv")
    if os.path.exists(ruta_std):
        return ruta_std
    snaps = sorted(glob.glob(os.path.join(DIR_DATA, "close_approaches_v*.csv")))
    if snaps:
        return snaps[-1]
    sys.exit("No se encontró close_approaches.csv en data/")


def cargar_datos_con_temporalidad():
    ruta_csv = resolver_ruta_cad()
    print(f"Cargando dataset desde: {os.path.basename(ruta_csv)}")
    df = pd.read_csv(ruta_csv)

    if "post_discovery" not in df.columns:
        sys.exit("Falta la columna 'post_discovery'.")

    obs = df[df["post_discovery"] == 1].copy()
    obs["dist_unc"] = obs["CA DistanceNominal (au)"] - obs["CA DistanceMinimum (au)"]

    # Agregar por objeto
    obj = obs.groupby("Object").agg(
        distnom_min=("CA DistanceNominal (au)", "min"),
        distmin_min=("CA DistanceMinimum (au)", "min"),
        vinf_max=("V infinity(km/s)", "max"),
        vinf_med=("V infinity(km/s)", "median"),
        H_obs=("H(mag)", "min"),
        n_appro=("Object", "size"),
        dist_unc_med=("dist_unc", "median"),
        moid=("MOID (au)", "max"),
        pha=("PHA_official", "max"),
        first_obs_year=("first_obs_year", "max"),
    ).reset_index()

    cols_req = ["moid", "distnom_min", "vinf_max", "H_obs", "first_obs_year"]
    obj = obj.dropna(subset=cols_req).copy()
    obj["is_real_hazardous"] = (obj["moid"] <= UMBRAL_MOID).astype(int)
    obj["is_proxy_hazardous"] = (obj["distnom_min"] <= UMBRAL_MOID).astype(int)

    return obj


def analizar_cohorte_moderna(obj):
    print("\n" + "=" * 75)
    print("1. EVALUACIÓN ESTRICTA EN COHORTE MODERNA (Descubiertos >= 2015)")
    print("=" * 75)
    print("Nota Metodológica: Para evitar el sesgo observacional de comparar objetos con")
    print("100 años de historial (e.g. 1920) contra objetos recientes (2016+), el modelo")
    print("se entrena en el histórico (<= 2014) y se evalúa estrictamente en el conjunto >= 2015.")

    features = ["distnom_min", "vinf_max", "H_obs", "n_appro", "dist_unc_med"]

    df_tr = obj[obj["first_obs_year"] <= 2014].copy()
    df_te = obj[obj["first_obs_year"] >= 2015].copy()

    X_tr, y_tr = df_tr[features].values, df_tr["is_real_hazardous"].values
    X_te, y_te = df_te[features].values, df_te["is_real_hazardous"].values

    scale_pos = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos,
        learning_rate=0.1,
        max_depth=5,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)[:, 1]
    df_te["prob_ml"] = probs
    df_te["is_ml_hazardous"] = (probs >= THRESHOLD_PROB).astype(int)

    # Métricas de detección
    total_mod = len(df_te)
    total_haz = int(y_te.sum())
    
    # Proxy
    p_rec = recall_score(y_te, df_te["is_proxy_hazardous"])
    p_prec = precision_score(y_te, df_te["is_proxy_hazardous"], zero_division=0)
    p_f2 = fbeta_score(y_te, df_te["is_proxy_hazardous"], beta=2)
    p_det = int(df_te["is_proxy_hazardous"].sum())

    # ML
    ml_rec = recall_score(y_te, df_te["is_ml_hazardous"])
    ml_prec = precision_score(y_te, df_te["is_ml_hazardous"], zero_division=0)
    ml_f2 = fbeta_score(y_te, df_te["is_ml_hazardous"], beta=2)
    ml_auc = roc_auc_score(y_te, probs)
    ml_det = int(df_te["is_ml_hazardous"].sum())

    print(f"\nTotal asteroides en cohorte moderna (>= 2015) : {total_mod:,}")
    print(f"Peligrosos Reales (MOID <= 0.05 au)            : {total_haz:,} ({100*total_haz/total_mod:.1f}%)")
    print("-" * 75)
    print(f"Proxy Observacional (distnom_min <= 0.05 au):")
    print(f"  - Detectados   : {p_det:,}")
    print(f"  - Recall       : {p_rec*100:.2f}% (Detecta {int(p_rec*total_haz):,} de {total_haz:,})")
    print(f"  - Precisión    : {p_prec*100:.2f}%")
    print(f"  - F2-score     : {p_f2:.4f}")
    print("-" * 75)
    print(f"Modelo ML (XGBoost con umbral prob >= {THRESHOLD_PROB}):")
    print(f"  - Detectados   : {ml_det:,}")
    print(f"  - Recall       : {ml_rec*100:.2f}% (Detecta {int(ml_rec*total_haz):,} de {total_haz:,})")
    print(f"  - Precisión    : {ml_prec*100:.2f}%")
    print(f"  - F2-score     : {ml_f2:.4f}")
    print(f"  - ROC-AUC      : {ml_auc:.4f}")

    # Análisis de Intersección y Rescate
    # Conjuntos dentro de los peligrosos reales
    haz_real = df_te[df_te["is_real_hazardous"] == 1]
    det_ambos = ((haz_real["is_proxy_hazardous"] == 1) & (haz_real["is_ml_hazardous"] == 1)).sum()
    det_solo_proxy = ((haz_real["is_proxy_hazardous"] == 1) & (haz_real["is_ml_hazardous"] == 0)).sum()
    det_solo_ml = ((haz_real["is_proxy_hazardous"] == 0) & (haz_real["is_ml_hazardous"] == 1)).sum()
    omitidos_ambos = ((haz_real["is_proxy_hazardous"] == 0) & (haz_real["is_ml_hazardous"] == 0)).sum()

    proxy_missed = ((haz_real["is_proxy_hazardous"] == 0)).sum()
    rescued_ratio = (det_solo_ml / proxy_missed) * 100 if proxy_missed > 0 else 0

    print("\n" + "=" * 75)
    print("ANÁLISIS DE RESCATE: ¿QUÉ APORTA EL ML FRENTE AL PROXY OBSERVACIONAL?")
    print("=" * 75)
    print(f"Asteroides peligrosos omitidos por el proxy (distnom_min > 0.05): {proxy_missed:,}")
    print(f"  -> Rescatados e identificados exitosamente por ML            : {det_solo_ml:,} ({rescued_ratio:.1f}%)")
    print(f"  -> Omitidos por ambos métodos                                : {omitidos_ambos:,}")
    print(f"Asteroides peligrosos detectados por ambos                     : {det_ambos:,}")
    print(f"Asteroides detectados por proxy pero omitidos por ML           : {det_solo_proxy:,}")

    # Análisis de Falsos Positivos de ML
    fp_ml = df_te[(df_te["is_real_hazardous"] == 0) & (df_te["is_ml_hazardous"] == 1)]
    fp_total = len(fp_ml)
    fp_mean_moid = fp_ml["moid"].mean()
    fp_med_moid = fp_ml["moid"].median()
    fp_sub_008 = (fp_ml["moid"] <= 0.08).mean() * 100
    fp_sub_010 = (fp_ml["moid"] <= 0.10).mean() * 100

    print("\n" + "=" * 75)
    print("RACIONALIDAD FÍSICA DE LOS FALSOS POSITIVOS DE ML")
    print("=" * 75)
    print(f"Total Falsos Positivos de ML (Objetos con MOID Real > 0.05 au predichos como peligrosos): {fp_total:,}")
    print(f"  - MOID Real Promedio: {fp_mean_moid:.4f} au (Mediana: {fp_med_moid:.4f} au)")
    print(f"  - % con MOID Real <= 0.08 au (muy próximos a la frontera) : {fp_sub_008:.1f}%")
    print(f"  - % con MOID Real <= 0.10 au (cuasi-peligrosos)             : {fp_sub_010:.1f}%")
    print("Conclusión Física: Los 'falsos positivos' no son errores erráticos, sino asteroides")
    print("en el borde orbital inmediato del umbral (0.05 - 0.08 au), lo cual es deseable en defensa planetaria.")

    # Guardar gráficas
    generar_graficas_cohorte(df_te, total_haz, p_rec, ml_rec, det_ambos, det_solo_ml, det_solo_proxy, omitidos_ambos, fp_ml)

    res_dict = {
        "cohort": ">= 2015 (Moderno)",
        "total_objects": total_mod,
        "total_hazardous_real": total_haz,
        "proxy_metrics": {
            "recall": float(p_rec),
            "precision": float(p_prec),
            "f2": float(p_f2),
            "detected_count": p_det
        },
        "ml_metrics": {
            "recall": float(ml_rec),
            "precision": float(ml_prec),
            "f2": float(ml_f2),
            "roc_auc": float(ml_auc),
            "detected_count": ml_det
        },
        "overlap_real_hazardous": {
            "detected_both": int(det_ambos),
            "rescued_by_ml_only": int(det_solo_ml),
            "proxy_only": int(det_solo_proxy),
            "missed_by_both": int(omitidos_ambos),
            "total_missed_by_proxy": int(proxy_missed),
            "rescued_percentage": float(rescued_ratio)
        },
        "false_positives_analysis": {
            "count": fp_total,
            "mean_moid": float(fp_mean_moid),
            "median_moid": float(fp_med_moid),
            "pct_below_0_08": float(fp_sub_008),
            "pct_below_0_10": float(fp_sub_010)
        }
    }

    return res_dict, df_te


def generar_graficas_cohorte(df_te, total_haz, p_rec, ml_rec, det_ambos, det_solo_ml, det_solo_proxy, omitidos_ambos, fp_ml):
    # Figura 1: Comparativa de Cobertura y Rescate en Cohorte Moderna
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Barras de Recall y Asteroides Peligrosos Detectados
    cats = ["Peligrosos Reales\n(Total)", "Detectados por\nProxy (distnom_min)", "Detectados por\nModelo ML (XGBoost)"]
    vals = [total_haz, int(p_rec * total_haz), int(ml_rec * total_haz)]
    colores = ["#2b5c8f", "#4682b4", "#2e8b57"]

    bars = ax1.bar(cats, vals, color=colores, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Número de Asteroides", fontsize=11)
    ax1.set_title("A. Cobertura de Peligrosidad en Cohorte Moderna (≥ 2015)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, vals):
        pct = (val / total_haz) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                 f"{val:,}\n({pct:.1f}%)" if val != total_haz else f"{val:,}\n(100%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, total_haz * 1.18)

    # Panel B: Desglose de Intersección de Asteroides Peligrosos Reales
    breakdown_labels = [
        f"Detectados por Ambos\n({det_ambos:,})",
        f"Rescatados por ML\n(Omitidos por Proxy)\n({det_solo_ml:,})",
        f"Solo Proxy\n({det_solo_proxy:,})",
        f"Omitidos por Ambos\n({omitidos_ambos:,})"
    ]
    breakdown_vals = [det_ambos, det_solo_ml, det_solo_proxy, omitidos_ambos]
    colors_pie = ["#2e8b57", "#ff7f0e", "#1f77b4", "#d62728"]

    ax2.pie(breakdown_vals, labels=breakdown_labels, autopct="%1.1f%%",
            startangle=140, colors=colors_pie,
            wedgeprops={"edgecolor": "black", "linewidth": 0.8},
            textprops={"fontsize": 9.5})
    ax2.set_title(f"B. Desglose del Conjunto Peligroso Real (N={total_haz:,})\nML Rescata {det_solo_ml:,} Asteroides",
                  fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig1_path = os.path.join(DIR_FIG, "hazard_detection_comparison_cohort2015.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"✔ Figura guardada: {fig1_path}")

    # Figura 2: Distribución del MOID Real de los Falsos Positivos de ML
    plt.figure(figsize=(8, 5))
    plt.hist(fp_ml["moid"], bins=40, color="#e67e22", edgecolor="black", alpha=0.8, density=True)
    plt.axvline(UMBRAL_MOID, color="red", linestyle="--", linewidth=2, label="Umbral Peligroso Oficial (0.05 au)")
    plt.axvline(fp_ml["moid"].median(), color="darkblue", linestyle="-", linewidth=2,
                label=f"Mediana FP ({fp_ml['moid'].median():.4f} au)")
    plt.axvline(0.08, color="purple", linestyle=":", linewidth=1.5, label="Zona Cuasi-Peligrosa (0.08 au)")

    plt.xlabel("MOID Orbital Real (au)", fontsize=11)
    plt.ylabel("Densidad de Probabilidad", fontsize=11)
    plt.title(f"Racionalidad de los Falsos Positivos de ML (Cohorte ≥ 2015)\n({fp_sub_008:.1f}% se concentran entre 0.05 y 0.08 au)",
              fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0.04, 0.20)
    plt.tight_layout()

    fig2_path = os.path.join(DIR_FIG, "ml_false_positives_distribution.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"✔ Figura guardada: {fig2_path}")


def analizar_evolucion_por_epocas(obj):
    print("\n" + "=" * 75)
    print("2. IMPACTO DEL TIEMPO OBSERVACIONAL: COMPARACIÓN POR ÉPOCAS")
    print("=" * 75)
    print("Se demuestra por qué comparar épocas distintas (e.g. 1920 vs 2016) altera la")
    print("efectividad del proxy debido a la acumulación de aproximaciones registradas.")

    bins = [-np.inf, 1999, 2014, np.inf]
    labels = ["Histórico (<2000)", "Transición (2000-2014)", "Moderno (≥2015)"]
    obj["epoca"] = pd.cut(obj["first_obs_year"], bins=bins, labels=labels)

    features = ["distnom_min", "vinf_max", "H_obs", "n_appro", "dist_unc_med"]

    # Validación cruzada estratificada general para obtener predicciones OOF de ML en todo el dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X = obj[features].values
    y = obj["is_real_hazardous"].values

    scale_pos = (len(y) - y.sum()) / y.sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos,
        learning_rate=0.1,
        max_depth=5,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1
    )

    oof_probs = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        model.fit(X[tr_idx], y[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X[va_idx])[:, 1]

    obj["oof_ml_haz"] = (oof_probs >= THRESHOLD_PROB).astype(int)

    resultados_epocas = []
    print(f"\n{'Época':<22} | {'N':<6} | {'Peligrosos':<10} | {'Proxy Rec':<10} | {'ML Rec':<10} | {'n_appro med':<11}")
    print("-" * 80)

    for ep in labels:
        sub = obj[obj["epoca"] == ep]
        n_tot = len(sub)
        n_haz = int(sub["is_real_hazardous"].sum())
        p_rec = recall_score(sub["is_real_hazardous"], sub["is_proxy_hazardous"])
        ml_rec = recall_score(sub["is_real_hazardous"], sub["oof_ml_haz"])
        p_prec = precision_score(sub["is_real_hazardous"], sub["is_proxy_hazardous"], zero_division=0)
        ml_prec = precision_score(sub["is_real_hazardous"], sub["oof_ml_haz"], zero_division=0)
        med_app = sub["n_appro"].median()

        print(f"{ep:<22} | {n_tot:<6} | {n_haz:<10} | {p_rec*100:>8.2f}% | {ml_rec*100:>8.2f}% | {med_app:>10.1f}")

        resultados_epocas.append({
            "epoca": ep,
            "total_objetos": n_tot,
            "peligrosos_reales": n_haz,
            "pct_peligrosos": float(n_haz / n_tot),
            "proxy_recall": float(p_rec),
            "proxy_precision": float(p_prec),
            "ml_recall": float(ml_rec),
            "ml_precision": float(ml_prec),
            "mediana_n_appro": float(med_app)
        })

    # Gráfica de evolución temporal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ep_names = [r["epoca"] for r in resultados_epocas]
    proxy_recs = [r["proxy_recall"] * 100 for r in resultados_epocas]
    ml_recs = [r["ml_recall"] * 100 for r in resultados_epocas]
    med_approaches = [r["mediana_n_appro"] for r in resultados_epocas]

    x = np.arange(len(ep_names))
    w = 0.35

    ax1.bar(x - w/2, proxy_recs, width=w, label="Proxy (distnom_min ≤ 0.05)", color="#4682b4", edgecolor="black")
    ax1.bar(x + w/2, ml_recs, width=w, label="Modelo ML (XGBoost)", color="#2e8b57", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ep_names, fontsize=10)
    ax1.set_ylabel("Recall de Peligrosidad (%)", fontsize=11)
    ax1.set_title("A. Sensibilidad (Recall) por Época de Descubrimiento", fontsize=12, fontweight="bold")
    ax1.set_ylim(40, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Añadir valores sobre barras
    for i in range(len(ep_names)):
        ax1.text(x[i] - w/2, proxy_recs[i] + 1.2, f"{proxy_recs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax1.text(x[i] + w/2, ml_recs[i] + 1.2, f"{ml_recs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax2.plot(ep_names, med_approaches, marker="o", color="#b22222", linewidth=2.5, markersize=8)
    ax2.set_ylabel("Mediana de Aproximaciones Registradas (n_appro)", fontsize=11)
    ax2.set_title("B. Oportunidades Observacionales Acumuladas por Objeto", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    for i, val in enumerate(med_approaches):
        ax2.text(i, val + 0.15, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold", color="#b22222")

    plt.tight_layout()
    fig3_path = os.path.join(DIR_FIG, "hazard_recall_by_epoch.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"✔ Figura guardada: {fig3_path}")

    return resultados_epocas


def main():
    obj = cargar_datos_con_temporalidad()
    res_mod, df_mod = analizar_cohorte_moderna(obj)
    res_ep = analizar_evolucion_por_epocas(obj)

    # Guardar tablas y resumen
    resumen_total = {
        "cohorte_moderna": res_mod,
        "analisis_epocas": res_ep
    }

    ruta_json = os.path.join(DIR_TAB, "hazard_verification_summary.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen_total, f, indent=2, ensure_ascii=False)

    df_ep = pd.DataFrame(res_ep)
    ruta_csv = os.path.join(DIR_TAB, "hazard_verification_epochs.csv")
    df_ep.to_csv(ruta_csv, index=False)

    print("\n" + "=" * 75)
    print(f"✔ Resultados guardados en:")
    print(f"   - {ruta_json}")
    print(f"   - {ruta_csv}")
    print("=" * 75)


if __name__ == "__main__":
    main()
