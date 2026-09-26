"""Predicción del MOID Orbital (Regresión y Clasificación Binaria).

Este script evalúa la capacidad de predecir el MOID (Minimum Orbit Intersection Distance)
y clasificar la condición geométrica MOID <= 0.05 au a partir de características
observacionales tempranas de aproximaciones cercanas (post_discovery == 1).

Uso (desde la raíz del repo):
    python scripts/predict_moid.py
"""

import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    fbeta_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.model_selection import KFold, StratifiedKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Asegurar codificación utf-8 en Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings('ignore')

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(RAIZ, "data")
DIR_FIG = os.path.join(RAIZ, "results", "figures")
DIR_TAB = os.path.join(RAIZ, "results", "tables")

os.makedirs(DIR_FIG, exist_ok=True)
os.makedirs(DIR_TAB, exist_ok=True)

SEED = 42
UMBRAL_MOID = 0.05
THRESHOLD_PROB = 0.35
FEATURES = ['distnom_min', 'vinf_max', 'H_obs', 'n_appro', 'dist_unc_med']

AZUL    = '#2b5c8f'
VERDE   = '#2e8b57'
NARANJA = '#e67e22'
ROJO    = '#c0392b'
VIOLETA = '#8e44ad'
GRIS    = '#7f8c8d'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f9f9f9',
    'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5, 'font.size': 11,
})


def cargar_y_agregar():
    ruta_std = os.path.join(DIR_DATA, 'close_approaches.csv')
    snaps    = sorted(glob.glob(os.path.join(DIR_DATA, 'close_approaches_v*.csv')))
    ruta_csv = ruta_std if os.path.exists(ruta_std) else (snaps[-1] if snaps else None)
    if ruta_csv is None:
        raise FileNotFoundError('No se encontró close_approaches.csv en data/')

    print(f'Cargando: {os.path.basename(ruta_csv)}')
    df_raw = pd.read_csv(ruta_csv)

    obs = df_raw[df_raw['post_discovery'] == 1].copy()
    obs['dist_unc'] = obs['CA DistanceNominal (au)'] - obs['CA DistanceMinimum (au)']
    obj = obs.groupby('Object').agg(
        distnom_min   = ('CA DistanceNominal (au)', 'min'),
        distmin_min   = ('CA DistanceMinimum (au)', 'min'),
        vinf_max      = ('V infinity(km/s)', 'max'),
        vinf_med      = ('V infinity(km/s)', 'median'),
        H_obs         = ('H(mag)', 'min'),
        n_appro       = ('Object', 'size'),
        dist_unc_med  = ('dist_unc', 'median'),
        moid          = ('MOID (au)', 'max'),
        pha           = ('PHA_official', 'max'),
        first_obs_year= ('first_obs_year', 'max'),
    ).reset_index()
    obj = obj.dropna(subset=['moid', 'distnom_min', 'vinf_max', 'H_obs']).copy()
    obj['is_moid_hazardous']  = (obj['moid'] <= UMBRAL_MOID).astype(int)
    obj['is_proxy_hazardous'] = (obj['distnom_min'] <= UMBRAL_MOID).astype(int)
    
    n = len(obj)
    ph = obj['is_moid_hazardous'].sum()
    print(f'Objetos procesados: {n:,} | PHAs reales (MOID<=0.05): {ph:,} ({100*ph/n:.1f}%)')
    return obj


def evaluar_regresion_moid(obj):
    print("\n" + "=" * 70)
    print("1. REGRESIÓN CONTINUA DEL MOID (Target: 'moid' en UA)")
    print("=" * 70)

    X = obj[FEATURES].values
    y = obj['moid'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    r2_naive  = r2_score(y, obj['distnom_min'].values)
    mae_naive = mean_absolute_error(y, obj['distnom_min'].values)
    print(f'Baseline ingenuo (distnom_min): R2={r2_naive:.4f} | MAE={mae_naive:.4f} au')

    if HAS_XGB:
        modelo_gb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                      subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
        gb_name = 'XGBoost'
    else:
        modelo_gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED)
        gb_name = 'GradientBoosting'

    modelos_reg = {
        'Ridge'         : Ridge(alpha=1.0),
        'Random Forest' : RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        gb_name         : modelo_gb,
    }
    
    resultados_reg = {}
    preds_oof_reg  = {}

    for nombre, model in modelos_reg.items():
        r2l, mael, rmsel, oof = [], [], [], np.zeros(len(y))
        for tr, va in kf.split(X):
            model.fit(X[tr], y[tr])
            p = model.predict(X[va])
            oof[va] = p
            r2l.append(r2_score(y[va], p))
            mael.append(mean_absolute_error(y[va], p))
            rmsel.append(np.sqrt(mean_squared_error(y[va], p)))
        resultados_reg[nombre] = {'R2': float(np.mean(r2l)), 'MAE': float(np.mean(mael)), 'RMSE': float(np.mean(rmsel))}
        preds_oof_reg[nombre] = oof
        print(f'Modelo {nombre:<15}: R2={np.mean(r2l):.4f}±{np.std(r2l):.4f} | MAE={np.mean(mael):.4f} au')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y, preds_oof_reg[gb_name], alpha=0.25, s=10, color=AZUL, label=f'OOF {gb_name}')
    lim = y.max() * 1.02
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='Perfecta y=x')
    ax.axvline(UMBRAL_MOID, color=NARANJA, ls=':', lw=1.5, label='Umbral 0.05 au')
    ax.axhline(UMBRAL_MOID, color=NARANJA, ls=':', lw=1.5)
    ax.set_xlabel('MOID Real (au)'); ax.set_ylabel('MOID Predicho (au)')
    ax.set_title(f'Regresión MOID ({gb_name})\nR2={resultados_reg[gb_name]["R2"]:.3f} | MAE={resultados_reg[gb_name]["MAE"]:.4f} au')
    ax.legend()

    ax2 = axes[1]
    names = list(resultados_reg.keys())
    r2v   = [resultados_reg[n]['R2'] for n in names]
    xp    = np.arange(len(names))
    bars  = ax2.bar(xp, r2v, color=[AZUL, VERDE, NARANJA], edgecolor='black', width=0.5)
    ax2.axhline(r2_naive, color=ROJO, ls='--', lw=1.8, label=f'Baseline R2={r2_naive:.3f}')
    ax2.set_xticks(xp); ax2.set_xticklabels(names); ax2.set_ylim(0, 1)
    ax2.set_ylabel('R2 (5-fold CV)'); ax2.set_title('Comparativa de Modelos — R2'); ax2.legend()
    for bar, val in zip(bars, r2v):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(DIR_FIG, 'regression_moid_comparison.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Figura guardada en: {fig_path}")

    return resultados_reg, preds_oof_reg, gb_name, r2_naive, mae_naive


def evaluar_clasificacion_moid(obj):
    print("\n" + "=" * 70)
    print("2. CLASIFICACIÓN BINARIA DEL UMBRAL PELIGROSO (Target: MOID <= 0.05 au)")
    print("=" * 70)

    X = obj[FEATURES].values
    y_cls = obj['is_moid_hazardous'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    prec_proxy = precision_score(y_cls, obj['is_proxy_hazardous'].values, zero_division=0)
    rec_proxy  = recall_score(y_cls, obj['is_proxy_hazardous'].values)
    f2_proxy   = fbeta_score(y_cls, obj['is_proxy_hazardous'].values, beta=2)
    print(f'Proxy (distnom_min<=0.05): Prec={prec_proxy:.4f} | Recall={rec_proxy:.4f} | F2={f2_proxy:.4f}')

    if HAS_XGB:
        spw = (len(y_cls)-y_cls.sum())/y_cls.sum()
        modelo_cls_gb = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=spw, learning_rate=0.05,
                                           max_depth=5, subsample=0.8, random_state=SEED, n_jobs=-1)
        gb_cls_name = 'XGBoost'
    else:
        modelo_cls_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED)
        gb_cls_name = 'GradientBoosting'

    modelos_cls = {
        'Reg. Logistica': LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000),
        'Random Forest' : RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1),
        gb_cls_name     : modelo_cls_gb,
    }
    resultados_cls = {}
    probs_oof_cls  = {}

    for nombre, model in modelos_cls.items():
        f2l, precl, recl, rocl = [], [], [], []
        oof = np.zeros(len(y_cls))
        for tr, va in skf.split(X, y_cls):
            model.fit(X[tr], y_cls[tr])
            pr = model.predict_proba(X[va])[:, 1]
            oof[va] = pr
            pd_ = (pr >= THRESHOLD_PROB).astype(int)
            f2l.append(fbeta_score(y_cls[va], pd_, beta=2))
            precl.append(precision_score(y_cls[va], pd_, zero_division=0))
            recl.append(recall_score(y_cls[va], pd_))
            rocl.append(roc_auc_score(y_cls[va], pr))
        resultados_cls[nombre] = {'F2': float(np.mean(f2l)), 'Precision': float(np.mean(precl)),
                                  'Recall': float(np.mean(recl)), 'ROC_AUC': float(np.mean(rocl))}
        probs_oof_cls[nombre] = oof
        print(f'Modelo {nombre:<20}: F2={np.mean(f2l):.4f} | Prec={np.mean(precl):.4f} | Recall={np.mean(recl):.4f} | AUC={np.mean(rocl):.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colores_m = [AZUL, VERDE, NARANJA]
    for (nombre, probs), color in zip(probs_oof_cls.items(), colores_m):
        auc = resultados_cls[nombre]['ROC_AUC']
        fpr, tpr, _ = roc_curve(y_cls, probs)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{nombre} (AUC={auc:.3f})')
        prec_c, rec_c, _ = precision_recall_curve(y_cls, probs)
        axes[1].plot(rec_c, prec_c, color=color, lw=2, label=nombre)
        
    axes[0].plot([0,1],[0,1],'k--',lw=1.2,label='Azar')
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('Recall'); axes[0].legend()
    axes[0].set_title('Curva ROC — Clasificación MOID ≤ 0.05 au')
    axes[1].plot(rec_proxy, prec_proxy, 'r*', ms=12, label='Proxy observacional')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precisión'); axes[1].legend()
    axes[1].set_title('Curva Precision-Recall')
    
    plt.tight_layout()
    fig_path = os.path.join(DIR_FIG, 'moid_classification_roc_pr.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Figura guardada en: {fig_path}")

    return resultados_cls


def aplicar_restricciones_fisicas(obj, preds_oof_reg, gb_name):
    print("\n" + "=" * 70)
    print("3. RESTRICCIONES GEOMÉTRICAS (FÍSICA)")
    print("=" * 70)

    y = obj['moid'].values
    oof_libre = preds_oof_reg[gb_name].copy()
    distnom_arr = obj['distnom_min'].values
    
    viol_mask = oof_libre > distnom_arr
    n_viol = viol_mask.sum()
    pct_viol = n_viol / len(y) * 100
    print(f'Violaciones geométricas: {n_viol:,} ({pct_viol:.2f}%)')

    oof_rest = np.minimum(oof_libre, distnom_arr)
    r2_libre = r2_score(y, oof_libre)
    mae_libre = mean_absolute_error(y, oof_libre)
    r2_rest = r2_score(y, oof_rest)
    mae_rest = mean_absolute_error(y, oof_rest)
    mejora = (mae_libre - mae_rest) / mae_libre * 100
    
    print(f'Libre       : R2={r2_libre:.4f} | MAE={mae_libre:.4f} au')
    print(f'Restringido : R2={r2_rest:.4f} | MAE={mae_rest:.4f} au')
    print(f'Mejora MAE al forzar física: {mejora:.2f}%')

    modelo_full = (xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED, n_jobs=-1)
                   if HAS_XGB else
                   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED))
    modelo_full.fit(obj[FEATURES].values, y)
    importances = modelo_full.feature_importances_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(distnom_arr[~viol_mask], oof_libre[~viol_mask], alpha=0.2, s=8, color=AZUL, label='Posible')
    ax.scatter(distnom_arr[viol_mask],  oof_libre[viol_mask],  alpha=0.4, s=10, color=ROJO, label='Violación')
    mv = min(0.3, max(distnom_arr.max(), oof_libre.max()))
    ax.plot([0,mv],[0,mv],'k--',lw=2,label='Límite físico (y=x)')
    ax.set_xlim(0,mv); ax.set_ylim(0,mv)
    ax.set_xlabel('distnom_min (au)'); ax.set_ylabel('MOID Predicho (au)')
    ax.set_title(f'Restricción Geométrica ({pct_viol:.1f}% violaciones)'); ax.legend()

    ax2 = axes[1]
    idx = np.argsort(importances)
    ax2.barh(range(len(idx)), importances[idx], color=AZUL, edgecolor='black')
    ax2.set_yticks(range(len(idx))); ax2.set_yticklabels([FEATURES[i] for i in idx])
    ax2.set_xlabel('Importancia (Gain)'); ax2.set_title('Importancia de Features (Regresión)')
    plt.tight_layout()
    fig_path = os.path.join(DIR_FIG, 'moid_physics_validation.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Figura guardada en: {fig_path}")

    return {
        'Violaciones_pct': float(pct_viol),
        'R2_libre': float(r2_libre), 'MAE_libre': float(mae_libre),
        'R2_restringido': float(r2_rest), 'MAE_restringido': float(mae_rest),
    }


def evaluar_prueba_historica(obj, gb_name):
    print("\n" + "=" * 70)
    print("4. PRUEBA HISTÓRICA RETROSPECTIVA (< 2000 vs >= 2000)")
    print("=" * 70)

    ANIO_CORTE = 2000
    df_historicos = obj[obj['first_obs_year'] < ANIO_CORTE].copy()
    df_modernos   = obj[obj['first_obs_year'] >= ANIO_CORTE].copy()

    n_h, ph_h = len(df_historicos), df_historicos['is_moid_hazardous'].sum()
    n_m, ph_m = len(df_modernos), df_modernos['is_moid_hazardous'].sum()
    
    print(f'NEOs históricos (pre-{ANIO_CORTE}) [PRUEBA]: {n_h:,} (PHAs: {ph_h:,})')
    print(f'NEOs modernos  ({ANIO_CORTE}+)  [ENTRENAMIENTO]: {n_m:,} (PHAs: {ph_m:,})')

    X_mod = df_modernos[FEATURES].values
    y_mod = df_modernos['moid'].values
    X_hist = df_historicos[FEATURES].values
    y_hist = df_historicos['moid'].values

    reg_hist = (xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
                if HAS_XGB else
                GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=SEED))
    
    rf_hist = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    ridge_hist = Ridge(alpha=1.0)

    reg_hist.fit(X_mod, y_mod)
    rf_hist.fit(X_mod, y_mod)
    ridge_hist.fit(X_mod, y_mod)

    pred_gb    = reg_hist.predict(X_hist)
    pred_rf    = rf_hist.predict(X_hist)
    pred_ridge = ridge_hist.predict(X_hist)
    pred_ens   = (pred_gb + pred_rf + pred_ridge) / 3

    distnom_h  = df_historicos['distnom_min'].values
    pred_gb_r  = np.minimum(pred_gb,  distnom_h)
    pred_ens_r = np.minimum(pred_ens, distnom_h)

    modelos_h = {
        gb_name                   : pred_gb,
        'Random Forest'           : pred_rf,
        'Ridge'                   : pred_ridge,
        'Ensemble'                : pred_ens,
        f'{gb_name}+restricción'  : pred_gb_r,
        'Ensemble+restricción'    : pred_ens_r,
    }

    print(f'\n{"Modelo":<30} {"R2":>8} {"MAE (au)":>10} {"RMSE (au)":>11}')
    print('-' * 65)
    metricas_h = {}
    for nombre, preds in modelos_h.items():
        r2_h   = r2_score(y_hist, preds)
        mae_h  = mean_absolute_error(y_hist, preds)
        rmse_h = np.sqrt(mean_squared_error(y_hist, preds))
        metricas_h[nombre] = {'R2': float(r2_h), 'MAE': float(mae_h), 'RMSE': float(rmse_h)}
        print(f'  {nombre:<28} {r2_h:>8.4f} {mae_h:>10.4f} {rmse_h:>11.4f}')
    
    r2_base  = r2_score(y_hist, distnom_h)
    mae_base = mean_absolute_error(y_hist, distnom_h)
    print('-' * 65)
    print(f'  {"Baseline (distnom_min)":<28} {r2_base:>8.4f} {mae_base:>10.4f}')

    y_mod_cls = df_modernos['is_moid_hazardous'].values
    y_hist_cls = df_historicos['is_moid_hazardous'].values
    
    spw2 = (len(y_mod_cls) - y_mod_cls.sum()) / max(y_mod_cls.sum(), 1)
    cls_hist = (xgb.XGBClassifier(n_estimators=300, scale_pos_weight=spw2, learning_rate=0.05,
                                   max_depth=6, random_state=SEED, n_jobs=-1)
                if HAS_XGB else
                GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=SEED))
    
    cls_hist.fit(X_mod, y_mod_cls)
    probs_h = cls_hist.predict_proba(X_hist)[:, 1]
    preds_cls_h = (probs_h >= THRESHOLD_PROB).astype(int)

    f2_h   = fbeta_score(y_hist_cls, preds_cls_h, beta=2)
    prec_h = precision_score(y_hist_cls, preds_cls_h, zero_division=0)
    rec_h  = recall_score(y_hist_cls, preds_cls_h)
    n_phas_real = int(y_hist_cls.sum())
    n_phas_det  = int(preds_cls_h[y_hist_cls==1].sum())
    
    print(f'\nClasificación PHAs históricos: F2={f2_h:.4f} | Prec={prec_h:.4f} | Recall={rec_h:.4f}')
    print(f'Detectados: {n_phas_det} de {n_phas_real} PHAs históricos reales')

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    modelos_plot = [
        (gb_name,                pred_gb,    'A', AZUL),
        ('Random Forest',        pred_rf,    'B', VERDE),
        ('Ensemble',             pred_ens,   'C', VIOLETA),
        ('Ensemble+restricción', pred_ens_r, 'D', NARANJA),
    ]
    positions = [(0,0),(0,1),(0,2),(1,0)]

    for (nombre, preds, panel, color), (row, col) in zip(modelos_plot, positions):
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(y_hist, preds, alpha=0.35, s=15, color=color, edgecolors='none')
        lim = max(y_hist.max(), preds.max()) * 1.05
        ax.plot([0,lim],[0,lim],'r--',lw=2,label='Perfecta y=x')
        ax.axvline(UMBRAL_MOID, color='gray', ls=':', lw=1.2, alpha=0.8)
        ax.axhline(UMBRAL_MOID, color='gray', ls=':', lw=1.2, alpha=0.8)
        ax.set_xlabel('MOID Real (au)', fontsize=10); ax.set_ylabel('MOID Predicho (au)', fontsize=10)
        ax.set_title(f'{panel}: {nombre}\nR2={metricas_h[nombre]["R2"]:.3f} | MAE={metricas_h[nombre]["MAE"]:.4f} au', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    errores = {k: np.abs(y_hist - v) for k, v in modelos_h.items()}
    ax5.boxplot(list(errores.values()), labels=list(errores.keys()),
                patch_artist=True, boxprops=dict(facecolor='#d6eaf8', color='navy'),
                medianprops=dict(color='red', linewidth=2))
    ax5.axhline(mae_base, color=NARANJA, ls='--', lw=1.5, label=f'Baseline MAE={mae_base:.4f}')
    ax5.set_ylabel('Error Absoluto (au)'); ax5.legend(fontsize=9)
    ax5.set_title('E: Distribución del Error\n(NEOs Históricos)', fontsize=11, fontweight='bold')
    ax5.tick_params(axis='x', rotation=25)

    ax6 = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(y_hist_cls, preds_cls_h)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No PHA','PHA'], yticklabels=['No PHA','PHA'],
                ax=ax6, linewidths=0.5, linecolor='white')
    ax6.set_xlabel('Predicho'); ax6.set_ylabel('Real')
    ax6.set_title(f'F: Clasificación PHAs Históricos\nF2={f2_h:.3f} | Recall={rec_h:.3f} | Prec={prec_h:.3f}', fontsize=11, fontweight='bold')

    fig.suptitle('Prueba Histórica Retrospectiva — Predicción del MOID de NEOs Antiguos (pre-2000)\nEntrenado con NEOs modernos (2000+) sin ver los MOIDs históricos', fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(DIR_FIG, 'historical_retroactive_test.png'), dpi=200, bbox_inches='tight')
    plt.close()

    df_res = df_historicos[['Object','moid','first_obs_year','n_appro','H_obs','distnom_min','is_moid_hazardous']].copy()
    df_res['moid_pred_ens_rest'] = pred_ens_r
    df_res['error_abs_ens_rest'] = np.abs(df_res['moid'] - pred_ens_r)
    df_res['prob_hist'] = probs_h
    df_res['cls_hist'] = preds_cls_h
    bins = [0, 0.05, 0.10, 0.20, 0.50, float('inf')]
    labels = ['PHA (<=0.05)','Cuasi-PHA (0.05-0.10)','Cercano (0.10-0.20)','Medio (0.20-0.50)','Lejano (>0.50)']
    df_res['zona_moid'] = pd.cut(df_res['moid'], bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    zonas_data = df_res.groupby('zona_moid', observed=True)['error_abs_ens_rest'].apply(list)
    ax.boxplot(list(zonas_data.values), labels=[str(z) for z in zonas_data.index],
               patch_artist=True, boxprops=dict(facecolor='#d5e8f8', color='navy'),
               medianprops=dict(color='red', lw=2.5), flierprops=dict(marker='.', ms=4, alpha=0.5))
    ax.set_ylabel('Error Absoluto MOID (au)'); ax.set_title('Distribución del Error por Zona Orbital (NEOs Históricos)')
    ax.set_xticklabels([str(z) for z in zonas_data.index], rotation=20, ha='right')

    ax2 = axes[1]
    sc = ax2.scatter(y_hist, pred_ens_r, c=df_historicos['n_appro'].values, cmap='viridis', alpha=0.5, s=18, edgecolors='none')
    lim2 = max(y_hist.max(), pred_ens_r.max()) * 1.05
    ax2.plot([0,lim2],[0,lim2],'r--',lw=2,label='Perfecta y=x')
    ax2.axvline(UMBRAL_MOID, color='orange', ls=':', lw=1.5, label='Umbral 0.05 au')
    ax2.axhline(UMBRAL_MOID, color='orange', ls=':', lw=1.5)
    plt.colorbar(sc, ax=ax2, label='n_appro')
    ax2.set_xlabel('MOID Real (au)'); ax2.set_ylabel('MOID Predicho (au)')
    ax2.set_title(f'MOID Real vs Predicho — NEOs Históricos (pre-2000)\nR2={metricas_h["Ensemble+restricción"]["R2"]:.3f} | MAE={metricas_h["Ensemble+restricción"]["MAE"]:.4f} au')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, 'historical_retroactive_final.png'), dpi=200)
    plt.close()

    df_res.to_csv(os.path.join(DIR_TAB, 'historical_retroactive_predictions.csv'), index=False)
    
    return {
        'n_neos_historicos': len(df_historicos),
        'n_phas_reales': n_phas_real,
        'Regresion': metricas_h,
        'Baseline': {'R2': float(r2_base), 'MAE': float(mae_base)},
        'Clasificacion': {'Recall': float(rec_h), 'Precision': float(prec_h), 'F2': float(f2_h), 'PHAs_detectados': n_phas_det},
    }


def main():
    obj = cargar_y_agregar()
    res_reg, preds_oof_reg, gb_name, r2_naive, mae_naive = evaluar_regresion_moid(obj)
    res_cls = evaluar_clasificacion_moid(obj)
    res_fisica = aplicar_restricciones_fisicas(obj, preds_oof_reg, gb_name)
    res_hist = evaluar_prueba_historica(obj, gb_name)

    resumen = {
        'Regresion_MOID_CV5': res_reg,
        'Clasificacion_MOID_CV5': res_cls,
        'Validacion_Fisica': res_fisica,
        'Prueba_Historica': res_hist,
    }

    ruta_json = os.path.join(DIR_TAB, "moid_full_analysis_summary.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✔ Resultados JSON guardados exitosamente en {ruta_json}")
    print(f"✔ Tabla CSV guardada en {os.path.join(DIR_TAB, 'historical_retroactive_predictions.csv')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
