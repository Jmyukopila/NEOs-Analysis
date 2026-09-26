# 06 — Predicción del MOID Orbital

[← Índice](README.md)

---

## Contexto Astronómico y Motivación

La definición oficial de la IAU y CNEOS (JPL/NASA) para un Asteroide Potencialmente Peligroso (PHA) requiere evaluar dos condiciones:

$$\text{PHA} \iff \text{MOID} \le 0.05\text{ au} \quad \wedge \quad H \le 22\text{ mag}$$

Mientras que la magnitud absoluta $H$ (o su estimador $H_{\text{obs}}$) puede derivarse con relativa rapidez tras las primeras fotometrías, el **MOID** (*Minimum Orbit Intersection Distance*) **no es un parámetro directamente observable** en un pasaje cercano. 

El MOID es una constante geométrica calculada a partir de los seis elementos orbitales completos de la órbita heliocéntrica del asteroide ($a, e, i, \Omega, \omega, M$). Obtener una solución orbital precisa exige acumular múltiples observaciones astrométricas distribuidas a lo largo de semanas, meses o años (un arco orbital `data_arc` prolongado).

Para un asteroide recién descubierto o con arcos observacionales cortos:
1. El **MOID oficial aún no está disponible** o presenta un alto margen de incertidumbre.
2. La distancia mínima observada (`distnom_min`) suele ser **significativamente mayor que el MOID** (mediana $3.8\times$ mayor entre los PHA reales), por lo que usar `distnom_min \le 0.05` como regla directa tiene un *recall* muy bajo ($\sim 29.5\%$).

Por ello, la **predicción del MOID mediante Machine Learning** busca modelar y estimar la distancia orbital teórica mínima a partir del conjunto de variables cinemáticas y observacionales disponibles tras los primeros avistamientos.

---

## Formulación de las Tareas de Machine Learning

El módulo de predicción (`scripts/predict_moid.py`) aborda dos problemas complementarios:

### 1. Regresión Continua del MOID
- **Variable Objetivo ($y$):** `MOID (au)` en escala continua.
- **Métricas:** Coeficiente de determinación ($R^2$), Error Absoluto Medio ($\text{MAE}$ en UA) y Raíz del Error Cuadrático Medio ($\text{RMSE}$ en UA).
- **Línea de Base (Baseline Naïve):** Asumir directamente que $\widehat{\text{MOID}} = \mathrm{distnom\_min}$.

### 2. Clasificación Binaria del Umbral Peligroso
- **Variable Objetivo ($y$):** `is_moid_hazardous` $= 1$ si $\text{MOID} \le 0.05\text{ au}$, y $0$ en caso contrario.
- **Métricas:** $F_2\text{-score}$ (priorizando minimizar falsos negativos), Precisión, Recall, ROC-AUC y PR-AUC.
- **Línea de Base:** Regla directa $\mathrm{distnom\_min} \le 0.05\text{ au}$.

---

## Set de Características Utilizadas (Features)

Para evitar fugas de información y colinealidades triviales, se utilizan únicamente variables observadas agregadas por objeto sobre eventos posteriores al descubrimiento (`post_discovery == 1`):

| Feature | Descripción | Rol |
| :--- | :--- | :--- |
| `distnom_min` | Mínima distancia nominal observada (UA) | Proxy observacional de proximidad |
| `vinf_max` | Máxima velocidad relativa en el infinito ($\text{km/s}$) | Parámetro cinemático |
| `H_obs` | Mínima magnitud absoluta observada (mag) | Estimación de tamaño intrínseco |
| `n_appro` | Número de aproximaciones cercanas observadas | Exposición observacional |
| `dist_unc_med` | Incertidumbre observacional mediana ($\text{dist}_{\text{nom}} - \text{dist}_{\text{min}}$) | Indicador de calidad orbital |

---

## Resultados Empíricos

### 1. Regresión del MOID
Evaluado mediante **Validación Cruzada de 5 Folds (5-Fold CV)** sobre los objetos con aproximaciones observadas:

- **Baseline Naïve (`distnom_min` directo):**
  - $R^2 = 0.5046$
  - $\text{MAE} = 0.0261\text{ au}$
  - $\text{RMSE} = 0.0542\text{ au}$

- **Modelos de Machine Learning (Set `kin+size`):**
  - **Ridge Regression:** $R^2 \approx 0.7715$, $\text{MAE} \approx 0.0227\text{ au}$
  - **Random Forest Regressor:** $R^2 \approx 0.7966$, $\text{MAE} \approx 0.0195\text{ au}$
  - **XGBoost Regressor:** **$R^2 = 0.8057$**, **$\text{MAE} = 0.0194\text{ au}$**, $\text{RMSE} = 0.0339\text{ au}$

El modelo **XGBoost Regressor** reduce el error absoluto medio ($\text{MAE}$) significativamente respecto a la cota observada directa, corrigiendo en gran medida el sesgo donde $\mathrm{distnom\_min} > \text{MOID}$.

### 2. Clasificación Binaria del Umbral $\text{MOID} \le 0.05\text{ au}$

- **Proxy directo ($\mathrm{distnom\_min} \le 0.05$):**
  - Precisión: $0.999$ | Recall: $0.785$ | $F_2\text{-score}$: $0.820$
- **Regresión Logística:** $F_2 = 0.907$ | $\text{ROC-AUC} = 0.946$
- **Random Forest Classifier:** $F_2 = 0.910$ | $\text{ROC-AUC} = 0.960$
- **XGBoost Classifier:** **$F_2 = 0.898$** | Precisión: $0.936$ | Recall: $0.889$ | **$\text{ROC-AUC} = 0.963$**

ML logra elevar el *recall* del umbral orbital desde el $78.5\%$ (del proxy ingenuo) hasta más del **$91.7\%$** (con RF), identificando una porción sustancial de los objetos cuyo MOID es peligroso aunque no hayan sido capturados en un acercamiento extremadamente próximo.

### 3. Validación Temporal (Hold-out por Año de Descubrimiento)
Para evaluar la generalización sobre asteroides de reciente descubrimiento:
- **Entrenamiento:** Objetos descubiertos $\le 2014$.
- **Test:** Objetos descubiertos $\ge 2015$.

- **Resultados en Test:**
  - Clasificación XGBoost: $F_2 = 0.909$, $\text{ROC-AUC} = 0.965$.

Aunque se evidencia un ligero descenso atribuible al sesgo de selección observacional de los sondeos modernos, el modelo mantiene una alta capacidad de estimación.

---

## Validación de Peligrosidad y Racionalidad (Alineación Temporal)

Para validar si la predicción de peligrosidad ($\text{MOID} \le 0.05\text{ au}$) es correcta respecto a los objetivos de defensa planetaria y comparar justamente contra el proxy observacional ($\mathrm{distnom\_min} \le 0.05\text{ au}$), **es indispensable respetar la coherencia temporal**:

> [!IMPORTANT]
> **Consistencia de Período Observacional:** No es metodológicamente correcto comparar un asteroide descubierto en 1920 con uno de 2016. Un objeto de 1920 ha acumulado más de 100 años de oportunidades para registrar un encuentro muy cercano ($\mathrm{distnom\_min}$ bajo), mientras que un asteroide descubierto en 2016 solo cuenta con 1 o 2 acercamientos registrados. 

Para eliminar este sesgo, evaluamos el desempeño del proxy frente a Machine Learning dividiendo el análisis en cohortes temporales estrictas mediante `scripts/verify_hazard_prediction.py`.

### 1. Evaluación Estricta en la Cohorte Moderna ($\ge 2015$)
Entrenando el modelo en descubrimientos históricos ($\le 2014$) y evaluándolo exclusivamente en asteroides modernos ($\ge 2015$, $N = 26,525$ objetos):

| Método | Recall (Sensibilidad) | Precisión | $F_2$-Score | Peligrosos Detectados |
| :--- | :---: | :---: | :---: | :---: |
| **Proxy Observacional ($\mathrm{distnom\_min} \le 0.05$)** | $81.38\%$ | **$99.93\%$** | $0.8452$ | $13,726$ |
| **Modelo ML (XGBoost)** | **$90.26\%$** | $93.78\%$ | **$0.9095$** | **$16,224$** |

### 2. Análisis de Rescate: ¿Qué Aporta el ML?
En esta cohorte moderna hay **$16,856$ asteroides verdaderamente peligrosos** ($\text{MOID} \le 0.05\text{ au}$):
- El proxy observacional **omite a $3,139$ asteroides peligrosos** debido a que, en su breve ventana de observación reciente, pasaron a distancias como $0.08$ o $0.15\text{ au}$, aun teniendo órbitas que cruzan la de la Tierra a menos de $0.05\text{ au}$.
- **ML rescata a $1,498$ de estos asteroides ($47.72\%$ de los omitidos)**, detectando su peligro orbital a partir de su cinemática temprana.
- **Pérdida nula:** El modelo ML no omitió **ningún** asteroide que el proxy hubiese capturado ($0$ casos).

### 3. Racionalidad Física de los Falsos Positivos de ML
De los $1009$ falsos positivos arrojados por ML (objetos con $\text{MOID} > 0.05\text{ au}$ clasificados como peligrosos):
- **MOID real promedio:** $0.0728\text{ au}$ (Mediana: $0.0628\text{ au}$).
- **$76.3\%$** tienen un MOID real $\le 0.08\text{ au}$.
- **$87.7\%$** tienen un MOID real $\le 0.10\text{ au}$.

Esto demuestra que **los errores de ML son físicamente racionales**: no confunde asteroides del cinturón principal con peligrosos, sino que clasifica como "peligrosos" a objetos en el borde orbital inmediato del umbral, lo cual es el comportamiento deseado y conservador para la defensa planetaria.

### 4. Evolución por Épocas de Descubrimiento

Al comparar las tres grandes épocas del catálogo:

| Época | Rango de Años | $N$ | Peligrosos Reales | Recall Proxy | Recall ML | Mediana $n_{\mathrm{appro}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Histórico** | $< 2000$ | $906$ | $381$ ($42.1\%$) | $61.15\%$ | **$78.22\%$** | $2.0$ |
| **Transición** | $2000 - 2014$ | $9,974$ | $4,996$ ($50.1\%$) | $69.96\%$ | **$84.57\%$** | $2.0$ |
| **Moderno** | $\ge 2015$ | $26,525$ | $16,856$ ($63.5\%$) | $81.38\%$ | **$90.99\%$** | $1.0$ |

En la era moderna, la mediana de aproximaciones registradas es apenas $1.0$. Esto confirma la necesidad de Machine Learning: ante la falta de un historial de múltiples aproximaciones acumuladas, el modelo predice la geometría orbital global desde el primer encuentro registrado.

---

## Conclusiones Metodológicas

1. **Factibilidad de la Predicción:** Es posible estimar razonablemente el `MOID` orbital ($R^2 \approx 0.806$) usando solo características cinemáticas y observacionales tempranas, cerrando la brecha antes de contar con un arco orbital prolongado.
2. **Superación del Proxy Ingenuo con Racionalidad Física:** El modelo ML rescata el $47.72\%$ de los asteroides peligrosos que el proxy observacional omite en sondeos modernos, y sus falsos positivos son cuasi-peligrosos ($76.3\%$ con $\text{MOID} \le 0.08\text{ au}$).
3. **Consistencia Temporal Comprobada:** Al evaluar en cohortes homogéneas de tiempo, el modelo demuestra una superioridad consistente sin depender de sesgos de acumulación de datos históricos.
4. **Reproducibilidad:** Los scripts `scripts/predict_moid.py` y `scripts/verify_hazard_prediction.py` permiten regenerar todas las métricas, figuras y tablas en `results/`.

---

[← Volver al Índice](README.md)
