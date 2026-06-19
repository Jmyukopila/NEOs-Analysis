# NEO Analysis: Clasificación de peligrosidad consciente de la selección

## Proyecto de Análisis de Objetos Cercanos a la Tierra (NEOs)

Este repositorio estudia los Objetos Cercanos a la Tierra (NEOs) a partir de su
registro de **aproximaciones cercanas** (Close-Approach Data) de JPL/NASA. La pregunta
central es:

> **¿Se puede inferir el carácter potencialmente peligroso (PHA) de un NEO a partir
> únicamente de la cinemática de sus aproximaciones observadas — sin los elementos
> orbitales (MOID, a, e, i) que lo definen — y cómo distorsiona esa inferencia la
> función de selección observacional de 126 años de catálogo?**

A diferencia de la literatura habitual, que alimenta al modelo (proxies de) las mismas
variables que *definen* la etiqueta PHA (`MOID ≤ 0.05 au` y `H ≤ 22`) y obtiene una
exactitud casi perfecta de forma casi circular, aquí predecimos la peligrosidad desde lo
**observable** y exponemos esa circularidad.

## Colaboradores

- **Jasen Yukopila**
- **Dariem Garcia**
- **Carlos Toro**

## Estructura del pipeline

El análisis está en dos notebooks que se ejecutan **en orden**:

1. **`data/ProyectoNeoRework_data.ipynb`** — descarga las aproximaciones cercanas (CAD
   API) y el catálogo de NEOs (SBDB API), construye las etiquetas de peligrosidad y
   guarda `data/close_approaches.csv` (cache de 30 días).
2. **`notebooks/ProyectoNeoRework_ml.ipynb`** — lee el CSV y ejecuta el análisis.

## Características principales

### 1. Obtención y etiquetado de datos
- Descarga automática de aproximaciones cercanas (CAD) y de NEOs (SBDB) de JPL/NASA.
- Cache local del CSV con validación de antigüedad (30 días).
- **Etiqueta oficial** `PHA_official` (flag `pha` de la SBDB, *ground truth*) y
  **etiqueta proxy** `PHA_proxy` derivada solo de lo observado. El `MOID` oficial se
  usa solo para etiquetar/validar, **nunca** como predictor.

### 2. Análisis exploratorio (no supervisado)
- **PCA** (PC1+PC2 ≈ 77.8% de varianza); revela redundancia (PC5 ≈ 0%).
- **K-Means** (k=4) sobre las 5 dimensiones estandarizadas.
- **t-SNE** (openTSNE) con búsqueda de hiperparámetros vía Optuna.

### 3. Clasificación supervisada de peligrosidad (PHA)
- Unidad de análisis: el **objeto** (agregación de eventos por `Object`).
- Modelos: Regresión Logística, Random Forest, XGBoost; manejo de desbalance
  (`class_weight` / `scale_pos_weight`).
- Métricas apropiadas para clases desbalanceadas: **F2, ROC-AUC, PR-AUC** (no accuracy).
- **Análisis de circularidad:** una regla de dos umbrales iguala al ML (F2 ≈ 0.98).
- **Validación temporal:** entrenar pre-2010 / evaluar post-2010; la prevalencia de PHA
  cae de ~15% a ~1% (sesgo de selección de los sondeos).
- **Explicabilidad (SHAP):** la señal de peligro la aporta el tamaño (`H`/diámetro), no
  la cinemática.

## Ejecución rápida (headless)

Para regenerar todas las figuras y métricas sin abrir Jupyter:

```bash
python .claude/skills/run-neos-analysis/driver.py
```

Las figuras se escriben en `results/figures/`. Ver
`.claude/skills/run-neos-analysis/SKILL.md` para detalles.

## Instalación

```bash
git clone https://github.com/JasenovichYukopila/NEOs-Analysis.git
cd NEOs-Analysis
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Uso |
|---|---|
| `numpy`, `pandas`, `scipy` | Cómputo numérico y manipulación de datos |
| `scikit-learn` | PCA, K-Means, modelos supervisados, métricas |
| `xgboost` | Clasificador gradient boosting |
| `imbalanced-learn` | Manejo de clases desbalanceadas |
| `shap` | Explicabilidad del modelo |
| `openTSNE` | Reducción de dimensiones (t-SNE) |
| `optuna` | Búsqueda de hiperparámetros para t-SNE |
| `matplotlib` | Visualización |
| `requests` | Consumo de las APIs de JPL |
