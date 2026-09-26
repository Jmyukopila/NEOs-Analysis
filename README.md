# NEO Analysis: sustitución de medida y función de selección en catálogos de aproximaciones cercanas

## Proyecto de Análisis de Objetos Cercanos a la Tierra (NEOs)

Este repositorio estudia los Objetos Cercanos a la Tierra (NEOs) a partir de su
registro de **aproximaciones cercanas** (Close-Approach Data) de JPL/NASA. La pregunta
central es:

> **¿Puede la distancia mínima observada en las aproximaciones cercanas sustituir al
> MOID orbital que define formalmente a un asteroide potencialmente peligroso (PHA), y
> cómo distorsiona esa sustitución la función de selección observacional de 126 años de
> catálogo?**

Es deliberadamente una pregunta de **medida**, no de predicción. PHA no es un fenómeno
físico sino una definición administrativa (`MOID ≤ 0.05 au` **y** `H ≤ 22`): cualquier
clasificador alimentado con proxies de esas dos cantidades no predice nada, reconstruye
una definición. La literatura habitual hace exactamente eso y reporta exactitudes casi
perfectas de forma circular. Aquí el aprendizaje supervisado se usa como **instrumento de
medida**, y lo que se reporta es el error de sustitución y su dependencia de la selección
observacional.

## Colaboradores

- **Jasen Yukopila**
- **Dariem Garcia**
- **Carlos Toro**

## Conceptos

¿Qué es un PHA? ¿Y el MOID? ¿Qué significa cada columna del CSV o cada métrica del
notebook? Todo está explicado en **[`docs/`](docs/README.md)**:

- [Conceptos astronómicos](docs/01-conceptos-astronomicos.md) — NEO, PHA, MOID, magnitud H, albedo, au, `v_rel` vs `v_inf`
- [Columnas del dataset](docs/02-columnas-del-dataset.md) — significado, unidades y rol de cada campo
- [Conceptos de machine learning](docs/03-conceptos-ml.md) — PCA, K-Means, F2, PR-AUC, SHAP, circularidad, sesgo de selección
- [Fuentes de datos](docs/04-fuentes-de-datos.md) — CAD API vs SBDB API
- [**Discrepancias físicas y teóricas**](docs/05-discrepancias.md) — auditoría contra la
  documentación de JPL, la literatura y otros repos, con evidencia reproducible
  (`python scripts/verificar_discrepancias.py`)
- [**Predicción del MOID Orbital**](docs/06-prediccion-moid.md) — regresión continua y clasificación binaria del umbral de peligro MOID ($\text{MOID} \le 0.05\text{ au}$) (`python scripts/predict_moid.py`)

## Estructura del pipeline

El análisis está dividido en notebooks exploratorios y scripts de ejecución directa:

1. **`data/ProyectoNeoRework_data.ipynb`** — descarga las aproximaciones cercanas (CAD
   API, troceada por décadas con reintentos) y el catálogo de NEOs (SBDB API), construye
   las etiquetas y guarda `data/close_approaches.csv` (cache de 30 días).
2. **`notebooks/ProyectoNeoRework_ml.ipynb`** — ejecuta el análisis exploratorio (PCA, K-Means) y clasificaciones base.
3. **`notebooks/regresion_moid_y_prueba_historica.ipynb`** — ejecuta el análisis profundo de regresión, restricciones físicas y la simulación histórica pre-2000.
4. **`scripts/`** — scripts de validación y predicción como `predict_moid.py` (regresión unificada con restricciones físicas) y `verify_hazard_prediction.py` (verificación de peligrosidad temporal).

Ambos notebooks y `scripts/verificar_discrepancias.py` comparten el paquete
**`neos/`**, donde viven una sola vez las constantes y las operaciones comunes:

- `neos/constantes.py` — umbrales de la definición PHA (`MOID ≤ 0.05 au`, `H ≤ 22`),
  constantes físicas (albedo asumido, `au`, `μ⊕`), `dist-max`, la semilla
  `random_state = 20`, los conjuntos de features y las rutas del repo.
- `neos/datos.py` — descarga CAD/SBDB con cache, limpieza de fechas, imputación del
  diámetro desde `H`, etiqueta proxy, agregación a nivel objeto y estadísticos de
  población.
- `neos/graficos.py` — guardado uniforme de figuras en `results/figures/` y barras
  agrupadas.
- `neos/modelos.py` — los tres clasificadores, el CV repetido y las métricas.

Las rutas del paquete son absolutas (ancladas a la raíz del repo), así que las
figuras y los CSV se resuelven igual sea cual sea el directorio de trabajo.

## Características principales

### 1. Obtención y etiquetado de datos
- Descarga automática de aproximaciones cercanas (CAD) y de NEOs (SBDB) de JPL/NASA.
- **`dist-max=0.5` au explícito.** El valor por defecto de la CAD API es `0.05` au, que es
  exactamente el umbral de distancia de la definición PHA: dejarlo implícito censura la
  muestra en el umbral de la propia etiqueta y la vuelve casi tautológica. Es un fallo
  silencioso que afecta a trabajos publicados —
  ver [discrepancia A](docs/05-discrepancias.md#a--el-dataset-está-censurado-en-el-umbral-que-define-la-etiqueta).
- **Columna `post_discovery`.** El 78.1 % de los eventos del catálogo son integraciones
  numéricas hacia atrás, anteriores al descubrimiento del objeto. El análisis principal
  usa solo los realmente observados (`post_discovery == 1`); el efecto de incluir los
  retroactivos está cuantificado aparte, en
  [docs/05-discrepancias.md](docs/05-discrepancias.md#f--el-436--de-las-aproximaciones-observadas-son-anteriores-al-descubrimiento).
- **Etiqueta oficial** `PHA_official` (flag `pha` de la SBDB, *ground truth*) y
  **etiqueta proxy** `PHA_proxy` derivada solo de lo observado. El `MOID` oficial se
  usa solo para etiquetar/validar, **nunca** como predictor.

### 2. Análisis exploratorio (no supervisado)
- **PCA** sobre **tres cantidades independientes** (`dist`, `v_inf`, `H`). Se excluyen
  `Diameter` y `v_rel` por ser funciones deterministas de las otras: incluirlas degeneraba
  el espectro y producía una componente de varianza ≈ 0 por aritmética, no por física.
- **K-Means** (k=4) sobre las dimensiones estandarizadas, no sobre las coordenadas PCA.

### 3. Clasificación supervisada como instrumento de medida
- Unidad de análisis: el **objeto** (agregación de eventos por `Object`, solo eventos
  posteriores al descubrimiento).
- Modelos: Regresión Logística, Random Forest, XGBoost; manejo de desbalance
  (`class_weight` / `scale_pos_weight`).
- Métricas apropiadas para clases desbalanceadas: **F2, ROC-AUC, PR-AUC** (no accuracy).
- **Ya no hay circularidad trivial.** Con el dataset corregido, la regla de dos umbrales
  que antes igualaba al ML obtiene F2=0.343 — muy por debajo de XGBoost (F2=0.713,
  `kin+size`). La censura del dataset anterior forzaba artificialmente esa igualdad.
- **La cinemática sí aporta señal sobre el tamaño solo:** `kin+size` (F2=0.713) supera a
  `size-only` (F2=0.653) y ambos superan ampliamente a `kin-only` (F2=0.382).
- **Validación temporal (por fecha real de descubrimiento):** entrenar con objetos
  descubiertos antes de 2015 (prevalencia PHA 14.6%) y evaluar con los posteriores
  (2.6%): F2 cae de 0.713 (CV aleatoria) a 0.611, con precisión 0.388 y recall 0.714 —
  el coste del desplazamiento de prevalencia.
- **Resultado central — sustitución del MOID por la distancia observada:** r=0.869,
  precisión=0.985, recall=0.295. Entre los PHA reales, la distancia observada mediana es
  **3.8× su MOID** — rara vez se captura el paso más cercano posible. Esa brecha es
  física, no ruido de muestreo.

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
| `matplotlib` | Visualización |
| `requests` | Consumo de las APIs de JPL |
