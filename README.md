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

El análisis está dividido en un pipeline ejecutable, notebooks exploratorios y scripts de auditoría:

1. **`scripts/pipeline_moid_pha.py`** — el pipeline principal. Regresión de MOID con
   holdout temporal, clasificación de PHA con CV repetido y umbral anidado, ablación de
   features e importancia SHAP. Es el sustituto de `predict_moid.py` y
   `verify_hazard_prediction.py`, que evaluaban sobre datos distintos y cuyos resultados
   no se podían comparar entre sí.
2. **`data/ProyectoNeoRework_data.ipynb`** — descarga las aproximaciones cercanas (CAD
   API, troceada por décadas con reintentos) y el catálogo de NEOs (SBDB API), construye
   las etiquetas y congela el snapshot versionado.
3. **`notebooks/ProyectoNeoRework_ml.ipynb`** — análisis exploratorio (PCA, K-Means) y
   clasificaciones base.
4. **`notebooks/regresion_moid_y_prueba_historica.ipynb`** — regresión, restricciones
   físicas y la simulación histórica pre-2000.
5. **`scripts/verificar_discrepancias.py`** — auditoría contra la documentación de JPL,
   la literatura y otros repositorios.

Todo el código comparte el paquete **`neos/`**, donde cada operación vive una sola vez:

| Módulo | Contenido |
|---|---|
| `neos/constantes.py` | Umbrales PHA (`MOID ≤ 0.05 au`, `H ≤ 22`), constantes físicas, `dist-max`, semilla `RANDOM_STATE = 20`, conjuntos de features y rutas |
| `neos/semillas.py` | Semilla canónica, derivadas por componente y fijación de hilos |
| `neos/datos.py` | Carga, verificación SHA-256 del snapshot, agregación por objeto con corte temporal |
| `neos/etiquetas.py` | Las cuatro definiciones de etiqueta y por qué no son intercambiables |
| `neos/features.py` | Matriz de diseño, alias de columnas y auditoría de fugas por cohorte |
| `neos/modelos.py` | Imputación dentro del pipeline, pesos de desbalance y calibración de umbral |
| `neos/evaluacion.py` | Métricas, bootstrap, McNemar y línea base de PR-AUC |
| `neos/xai.py` | Preparación de matrices e importancia SHAP |
| `neos/graficos.py` | Guardado uniforme de figuras |

Las rutas del paquete son absolutas (ancladas a la raíz del repo), así que las
figuras y los CSV se resuelven igual sea cual sea el directorio de trabajo.

## Resultados verificados

Cifras de `python scripts/pipeline_moid_pha.py` sobre el snapshot
`data/close_approaches_v20260811.csv` (340 469 eventos, 74 530 post-descubrimiento),
con corte temporal en 2015 y 10 semillas (20–29).

**Regresión de MOID, holdout temporal** (10 063 objetos de entrenamiento, 26 528 de prueba):

| Métrica | Valor |
|---|---|
| MAE | 0.0174 au |
| RMSE | 0.0326 au |
| R² | 0.796 |

**Clasificación de PHA** (36 591 objetos, prevalencia 5.58 %, CV 5-fold × 10 semillas):

| Métrica | Valor |
|---|---|
| F2 | 0.717 ± 0.002 (entre semillas) |
| F2, IC 95 % bootstrap | [0.704, 0.728] |
| PR-AUC | 0.698 ± 0.003 |
| Línea base AP (prevalencia) | 0.056 |
| ROC-AUC | 0.970 |
| Recall / precisión @ umbral F2 | 0.93 / 0.37 |

**Ablación** — la comparación que decide si hay señal más allá de la distancia:

| Conjunto | Features | F2 | PR-AUC |
|---|---|---|---|
| `kin+size` | distancia + v∞ + H + nº | 0.709 | 0.687 |
| **`sin-dist`** | **todo menos la distancia** | **0.693** | **0.492** |
| `size-only` | H | 0.660 | 0.286 |
| `kin-only` | distancia + v∞ + nº | 0.373 | 0.165 |
| `kin-puro` | distancia + v∞ | 0.360 | 0.152 |
| `proxy-1feat` | distancia sola (regla de umbral) | 0.276 | 0.081 |

`sin-dist` supera a `proxy-1feat` por un factor de seis en PR-AUC: hay señal
cinemática y de tamaño independiente de la distancia observada. La distancia por sí
sola no reconstruye la regla.

**SHAP** (importancia media absoluta, `E[|SHAP|]`): para PHA, `H_obs` domina (4.86),
seguida de `vinf_closest` (1.58) y `distnom_min` (1.48). Para la regresión de MOID,
`distnom_min` es la feature que más pesa (0.048).

## Características principales

### 1. Obtención y etiquetado de datos
- Descarga automática de aproximaciones cercanas (CAD) y de NEOs (SBDB) de JPL/NASA.
- **Snapshot congelado y verificado.** `data/close_approaches_v20260811.csv` con su
  SHA-256 en `data/snapshot_info.json`. El pipeline verifica el hash antes de cargar y
  aborta si no coincide, en lugar de analizar en silencio un catálogo distinto.
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
- **Cuatro etiquetas con nombre propio.** `MOID ≤ 0.05` por separado **no** es un PHA:
  de 22 237 objetos que la cumplen en el catálogo post-descubrimiento, solo 2 246 son
  PHAs. Las 19 991 restantes están marcados por la geometría y les falta el criterio de
  tamaño. Confundir ambas dio lugar a un "accuracy" de 0.999 que no significaba nada.
- **Etiqueta oficial** `et_pha_oficial` (flag `pha` de la SBDB, *ground truth*) y
  **etiqueta proxy** `et_proxy` derivada solo de lo observado. El `MOID` oficial se
  usa solo para etiquetar/validar, **nunca** como predictor.
- **NaN se propaga como ausencia, no como 0.** Con `MOID` sin dato, `(NaN ≤ 0.05)` es
  `False` en numpy, así que la comparación directa etiquetaba como "no PHA" a los objetos
  de los que no se conoce la órbita.

### 2. Análisis exploratorio (no supervisado)
- **PCA** sobre **tres cantidades independientes** (`dist`, `v_inf`, `H`). Se excluyen
  `Diameter` y `v_rel` por ser funciones deterministas de las otras: incluirlas degeneraba
  el espectro y producía una componente de varianza ≈ 0 por aritmética, no por física.
- **K-Means** (k=4) sobre las dimensiones estandarizadas, no sobre las coordenadas PCA.

### 3. Clasificación supervisada como instrumento de medida
- Unidad de análisis: el **objeto** (agregación de eventos por `Object`, solo eventos
  posteriores al descubrimiento).
- Modelos: Regresión Logística, Random Forest, XGBoost; manejo de desbalance con factor
  explícito (`scale_pos_weight` siempre ≥ 1, nunca el inverso, que penalizaría a la clase
  minoritaria).
- Métricas apropiadas para clases desbalanceadas: **F2, PR-AUC, ROC-AUC** (no accuracy),
  siempre con la línea base de prevalencia al lado, para que el PR-AUC tenga un suelo
  con sentido.
- **El holdout temporal construye dos ventanas, no una.** Los acercamientos anteriores y
  posteriores al corte se agregan por separado y cada objeto conserva solo la ventana que
  le corresponde por su año de descubrimiento. Con una sola tabla filtrada por
  `hasta=corte` la mitad de prueba sale vacía; con datos a ambos lados, el mismo objeto
  aparecía dos veces en entrenamiento con features contradictorias. `comprobar_fugas_temporales()`
  aborta si alguna de las tres condiciones se incumple.
- **El umbral de decisión se elige en un fold de validación aparte**, nunca sobre la
  prueba: en la ablación se calculaba con el mismo conjunto sobre el que se reportaba el
  F2, así que cada número de la tabla estaba optimizado sobre sí mismo.
- **Cada número lleva su incertidumbre**, y se reportan las fuentes por separado porque
  no se suman: dispersión entre semillas, dispersión entre folds e intervalo bootstrap
  sobre las filas de prueba.

### 4. Reproducibilidad
- `neos/semillas.py` fija Python, numpy y el número de hilos de BLAS/OpenMP: XGBoost
  reduce en paralelo y esa reducción no es determinista.
- `scripts/comprobar_sintaxis.py` verifica que todos los scripts parsean e importan sin
  ejecutarse, antes de que un `NameError` aparezca al ejecutar con datos reales.

## Ejecución rápida (headless)

Para regenerar todas las figuras, tablas y métricas del pipeline principal:

```bash
python scripts/pipeline_moid_pha.py
```

Opciones útiles:

```bash
# Menos semillas y sin la parte lenta (SHAP)
python scripts/pipeline_moid_pha.py --semillas 5 --sin-shap

# Sin la ablación, que es la parte más cara
python scripts/pipeline_moid_pha.py --sin-ablation

# Otra etiqueta (por defecto: et_pha, la definición oficial)
python scripts/pipeline_moid_pha.py --etiqueta et_moid_le_005
```

Las salidas van a `results/figures/` y `results/tables/`, una tabla por bloque
(`regresion_moid.csv`, `clasificacion_pha.csv`, `ablation_features.csv`,
`shap_*_importance.csv`) y la configuración completa en `pipeline_config.json`.

Para regenerar el análisis exploratorio sin abrir Jupyter:

```bash
python .claude/skills/run-neos-analysis/driver.py
```

## Instalación

```bash
git clone https://github.com/JasenovichYukopila/NEOs-Analysis.git
cd NEOs-Analysis
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Versión | Uso |
|---|---|---|
| `numpy` | 2.5.3 | Cómputo numérico |
| `pandas` | 3.0.6 | Manipulación de datos |
| `scipy` | 1.16.2 | Estadística y tests binomiales |
| `scikit-learn` | 1.6.1 | Modelos supervisados, CV, métricas |
| `xgboost` | 3.2.0 | Clasificador gradient boosting |
| `shap` | 0.51.0 | Explicabilidad |
| `matplotlib` | 3.11.2 | Visualización |
| `seaborn` | 0.13.2 | Visualización |
| `requests` | 2.34.2 | Consumo de las APIs de JPL |

Entorno verificado: **Python 3.12.10**. Las versiones están fijadas con `==` porque
varias cifras cambian con la versión; si se actualiza alguna, hay que volver a ejecutar
el pipeline y regenerar los números de este README.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest              # bateria completa
python scripts/comprobar_sintaxis.py
```

Los tests **no tocan la red** y los que necesitan el snapshot lo saltan si no está
disponible. Cubren:

- **`neos/`** — la resolución de etiquetas (clave o columna), la propagación de NaN, el
  orden de las columnas de la matriz de diseño, que el factor de desbalance nunca baje
  de 1, que el imputador esté dentro del pipeline, que el bootstrap contenga el punto
  estimado, y que SHAP devuelva un `ndarray` y no un `Explanation`.
- **`scripts/pipeline_moid_pha.py`** — que el holdout temporal produzca dos mitades no
  vacías y disjuntas, que `i_in`/`i_val`/`i_te` no se solapen, que el umbral salga de
  validación y no de prueba, y que la CLI acepte las cuatro columnas de etiqueta.
- **`scripts/verificar_discrepancias.py`** — cada bloque A–I con un catálogo «sano» y con
  catálogos censurados que el script debe detectar.
- **Las funciones auxiliares de `data/ProyectoNeoRework_data.ipynb`**, extraídas con
  `ast` para poder ejecutarlas sin lanzar la descarga (ver `tests/conftest.py`).
- **Un guard de calidad de texto** (`tests/test_calidad_texto.py`) que detecta
  caracteres de alfabetos no latinos y palabras pegadas. No es cosmético: un carácter
  CJK colado en un docstring llegó a estar en el repositorio y rompió una figura.
