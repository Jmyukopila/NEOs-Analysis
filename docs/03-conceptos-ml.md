# 03 — Conceptos de machine learning

[← Índice](README.md)

Qué es cada técnica y cada métrica que aparece en `notebooks/ProyectoNeoRework_ml.ipynb`,
y **por qué** se eligió en este proyecto concreto.

---

## Estandarización (`StandardScaler`)

Transforma cada variable a media 0 y desviación típica 1: `z = (x − μ) / σ`.

Es obligatorio aquí porque las escalas son incomparables: las distancias están en
centésimas de au, las velocidades en decenas de km/s y H en unidades de magnitud. Sin
estandarizar, PCA y K-Means (que miden distancias euclídeas) quedarían dominados por la
variable de mayor rango numérico, que no es la más informativa sino simplemente la que
se mide en unidades más grandes.

---

## PCA (Análisis de Componentes Principales)

Rotación del espacio de features hacia ejes nuevos (**componentes principales**),
ordenados por cuánta varianza capturan. PC1 es la dirección de máxima dispersión de los
datos, PC2 la siguiente ortogonal a ella, etc.

Dos usos en el proyecto:

1. **Visualización** — proyectar las 5 dimensiones en 2 para poder dibujarlas
   (PC1+PC2 ≈ **77.8 %** de la varianza).
2. **Diagnóstico de redundancia** — que PC5 aporte ≈ **0 %** de varianza significa que
   hay una combinación lineal de las 5 features que es casi constante: las variables no
   son independientes. Esto anticipa el análisis de colinealidad.

> **Advertencia de interpretación:** los ejes PCA son combinaciones lineales, no
> magnitudes físicas. "PC1 alto" no significa nada por sí solo hasta mirar los *loadings*
> (los pesos de cada feature original en la componente).

---

## K-Means

Algoritmo de clustering: divide los datos en **k** grupos minimizando la distancia de
cada punto al centroide de su grupo. En el proyecto: `k = 4`, `n_init = 10`,
`random_state = 20`.

Dos decisiones deliberadas del notebook:

- **El clustering se hace en el espacio estandarizado completo, no sobre las coordenadas
  PCA.** El PCA se usa *solo para dibujar* el resultado. Agrupar sobre coordenadas
  reducidas descartaría información real; no hay que mover el `fit` de K-Means a las
  coordenadas reducidas.
- **`k = 4` está fijado.** La celda del método del codo es exploratoria: se conserva
  como justificación documental, no como selector automático.

### Método del codo

Se grafica la inercia (suma de distancias al cuadrado a los centroides) frente a k. La
inercia siempre baja al subir k; se busca el punto donde deja de bajar bruscamente — el
"codo" — como compromiso entre ajuste y parsimonia. Es un criterio visual, no una prueba
estadística.

---

## Colinealidad y VIF

Dos features son **colineales** cuando una es prácticamente predecible desde la otra.
Consecuencias: coeficientes inestables en modelos lineales, importancias repartidas
arbitrariamente entre variables redundantes, e interpretación engañosa.

Se diagnostica con la matriz de correlaciones (se reportan los pares con `|corr| > 0.9`)
y con el **VIF** (*Variance Inflation Factor*), que mide cuánto se infla la varianza del
coeficiente de una variable por su correlación con las demás.

En este dataset hay dos fuentes claras de redundancia:

- `Diameter(km)` fue **imputado desde** `H(mag)` → dependencia funcional exacta en esas filas.
- `v_rel`, `v_inf` y sus agregados son casi idénticos en pasos lejanos.

La **poda** resultante conserva una sola distancia (`distnom_min`), una sola velocidad
(`vrel_max`) y un solo indicador de tamaño (`H_obs`).

---

## Desbalance de clases

Solo el **7.3 %** de los objetos son PHA (1 379 de 18 927 — 1 PHA por cada 12 no-PHA).
Un clasificador que prediga siempre "no peligroso" acertaría el 93 % de las veces sin
haber aprendido nada:
por eso **la accuracy se omite deliberadamente** de todo el reporte.

Cómo se maneja en el notebook:

- `class_weight="balanced"` en Regresión Logística y Random Forest — pondera cada clase
  inversamente a su frecuencia.
- `scale_pos_weight = n_neg / n_pos` en XGBoost — el equivalente en esa librería.

---

## Métricas

### Precisión y recall

- **Precisión** = de los que marqué como peligrosos, ¿qué fracción lo eran?
  Controla las **falsas alarmas**.
- **Recall** (sensibilidad) = de los peligrosos que existen, ¿qué fracción detecté?
  Controla los **peligros que se me escapan**.

### F2 (métrica principal del proyecto)

Media armónica ponderada de precisión y recall que da **4× más peso al recall**:

```
F_β = (1 + β²) · precisión · recall / (β² · precisión + recall)      con β = 2
```

Se eligió F2 y no F1 por el coste asimétrico del dominio: **no detectar un asteroide
peligroso es mucho peor que investigar de más uno inofensivo**. F1 (β=1) trataría ambos
errores como equivalentes, lo cual sería incorrecto aquí.

### PR-AUC (`average_precision`)

Área bajo la curva precisión–recall. Es la métrica de referencia con clases muy
desbalanceadas, porque su línea base es la prevalencia de la clase positiva (0.073 aquí):
no se deja engañar por la abundancia de negativos.

### ROC-AUC

Área bajo la curva ROC (tasa de verdaderos positivos vs falsos positivos). Se reporta
por convención, pero con desbalance fuerte tiende a dar valores optimistas — de ahí que
todos los modelos ronden 0.99 mientras el F2 sí discrimina entre ellos.

### Umbral de decisión

Los clasificadores devuelven una **probabilidad**, no una clase. Convertirla en decisión
exige un umbral, y `0.5` es una convención arbitraria, no un óptimo. El notebook calcula
probabilidades **fuera de fold** (`cross_val_predict` con `StratifiedKFold`) y reporta
el umbral que **maximiza F2**, en vez de asumir 0.5.

---

## Validación cruzada

### Repeated Stratified K-Fold (5 × 3)

Los datos se parten en 5 pliegues (*folds*); cada uno se usa una vez como test mientras
los otros 4 entrenan. **Estratificada**: cada pliegue conserva la proporción de PHA del
conjunto total — imprescindible con 7 % de positivos, o algún pliegue podría quedarse
casi sin ellos. **Repetida** 3 veces con particiones distintas, para reportar
media ± desviación típica y no un número que dependa de una partición afortunada.

**Unidad de análisis: el objeto**, no el evento. Si se hiciera CV a nivel de evento, el
mismo asteroide aparecería en entrenamiento y en test — fuga de información garantizada.

---

## Circularidad y *leakage*

**Leakage** (fuga): el modelo accede, directa o indirectamente, a información que no
estaría disponible en el momento de predecir — o directamente a la respuesta.

**Circularidad** es el caso extremo que este proyecto expone en la literatura. PHA se
*define* como `MOID ≤ 0.05 au ∧ H ≤ 22`. Buena parte de la literatura alimenta al modelo
con esas mismas variables (o proxies suyos) y obtiene exactitud casi perfecta,
presentándola como un logro predictivo. No lo es: el modelo está reconstruyendo una
definición, no descubriendo física.

El notebook lo contrasta con un **baseline de regla de dos umbrales** — aplicar los mismos
umbrales que definen PHA sobre las variables observadas. En este proyecto esa regla
alcanza **F2 = 0.343**, muy por debajo de XGBoost sobre `kin+size` (**F2 = 0.712**): la
brecha entre la regla ingenua y el modelo entrenado es justo la evidencia de que **aquí**
no hay circularidad — hay señal real que el ML captura y la regla no. (Una versión
anterior y censurada del dataset sí igualaba regla y modelo en F2 ≈ 0.98; ver
[docs/05-discrepancias.md](05-discrepancias.md#a--el-dataset-está-censurado-en-el-umbral-que-define-la-etiqueta)
para el porqué y la corrección.)

---

## SHAP (SHapley Additive exPlanations)

Método de explicabilidad basado en los **valores de Shapley** de teoría de juegos:
reparte la predicción entre las features atribuyendo a cada una su contribución marginal
promediada sobre todas las coaliciones posibles de features.

Ventaja sobre la importancia por impureza de un Random Forest: es consistente y da
atribuciones **por predicción individual**, no solo un ranking global.

Resultado en este proyecto: `H_obs` sigue siendo la variable individual más informativa
(F2 `size-only` = 0.653 vs F2 `kin-only` = 0.382), coherente con ser una de las dos
variables que definen PHA. Pero a diferencia de una versión censurada anterior del
dataset, la cinemática **sí** añade información por encima del tamaño solo:
`kin+size` (F2 = 0.713) supera a `size-only` en +0.06, un incremento real que el gráfico
de importancias y el *beeswarm* de SHAP desglosan.

---

## Sesgo de selección (*selection bias*)

El catálogo no es una muestra aleatoria del cielo: es el resultado de 126 años de
campañas de observación con capacidades muy distintas. Los sondeos antiguos solo
detectaban objetos grandes y cercanos — exactamente el perfil PHA.

El notebook lo cuantifica de dos formas:

### Función de selección temporal

Prevalencia de PHA por **década de primera observación real** (`first_obs_year`, no el
año del primer evento del catálogo, que puede ser retroactivo hasta 1900 — el 78.1 % de
los eventos del catálogo lo son). La prevalencia cae con el tiempo. En paralelo se mide
`corr(data_arc, PHA) = 0.351`: los PHA tienen arcos orbitales algo más largos porque se
les observa más, lo que es un **confundidor de caracterización**, no física.

### Hold-out temporal

La prueba honesta: entrenar solo con objetos **descubiertos antes de 2015** (10 880
objetos, prevalencia PHA 14.6 %) y evaluar con los descubiertos **después** (26 525
objetos, prevalencia 2.6 %). Las medianas de imputación se calculan **solo con el
train**, para que no se filtre información del futuro hacia el pasado.

Resultado con XGBoost `kin+size`:

| Escenario | F2 | Precisión | Recall |
|---|---|---|---|
| CV aleatoria | 0.713 | — | — |
| Hold-out temporal (<2015 → ≥2015) | 0.611 | **0.388** | 0.714 |

El recall se mantiene razonable, pero la precisión cae con fuerza. Esa caída es
principalmente **el coste del desplazamiento de prevalencia** (14.6 % → 2.6 %): en un
mundo donde los peligrosos son casi seis veces más raros, el mismo modelo genera muchas
más falsas alarmas por cada acierto. Una CV aleatoria nunca habría revelado esto.

### Sesgo de muestreo

Adicionalmente se compara la prevalencia PHA del dataset (condicionado a tener
aproximaciones registradas) contra el total de NEOs de la SBDB, **por cada condición de la
definición por separado** — la prevalencia conjunta puede ocultar sesgos que se cancelan:

| población | n | PHA | MOID≤0.05 | H≤22 |
|---|---:|---:|---:|---:|
| todos los NEOs (SBDB) | 42 318 | 6.1 % | 54.1 % | 27.6 % |
| en el catálogo CAD | 37 405 | 6.0 % | 59.4 % | 21.4 % |

El enriquecimiento en la prevalencia conjunta es efectivamente leve (6.1 % → 6.0 %), pero
no porque no haya sesgo por aproximación: es porque el catálogo enriquece la condición
MOID (+5.3 pp) y empobrece la condición H (−6.2 pp) en direcciones opuestas, y ambas
casi se cancelan en el producto.

---

## Reproducibilidad

`random_state = 20` se define **una sola vez** en el preámbulo y se propaga a todo lo
estocástico: `KMeans`, `PCA`/`StandardScaler`, los tres clasificadores y los esquemas de
CV. Al modificar el notebook, hay que seguir usando esa variable en vez de literales
sueltos.

---

[← Columnas del dataset](02-columnas-del-dataset.md) · [Índice](README.md) · [Siguiente: fuentes de datos →](04-fuentes-de-datos.md)
