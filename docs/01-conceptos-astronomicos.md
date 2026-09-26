# 01 — Conceptos astronómicos

[← Índice](README.md)

---

## NEO (Near-Earth Object)

Objeto del Sistema Solar — asteroide o cometa — cuya órbita lo acerca a la de la Tierra.
El criterio formal es que su **perihelio** (punto más cercano al Sol de su órbita) sea
menor que **1.3 au**. Es una definición puramente orbital: ser NEO no implica ser
peligroso, solo estar en el vecindario.

En este proyecto el universo de estudio son los NEOs con al menos **una aproximación
cercana registrada** en el catálogo de JPL entre 1900 y hoy. Eso ya es un filtro: no
todos los NEOs conocidos aparecen (ver *sesgo de muestreo* en
[03 — ML](03-conceptos-ml.md#sesgo-de-selección-selection-bias)).

---

## Aproximación cercana (*close approach*, CA)

Un **evento**, no un objeto: el instante en que un NEO pasa por su punto de máximo
acercamiento a un cuerpo (aquí, la Tierra) en un paso concreto. Un mismo asteroide
puede tener decenas de aproximaciones registradas a lo largo de un siglo.

Esta distinción es la clave de la estructura de los datos:

- El CSV bruto tiene **una fila por evento** (`Object` se repite).
- El análisis supervisado agrega a **una fila por objeto** (`groupby("Object")`),
  porque la etiqueta PHA es una propiedad del objeto, no del paso.

Mezclar ambos niveles causaría fuga de información entre entrenamiento y test (el mismo
asteroide en los dos lados del split).

---

## PHA (Potentially Hazardous Asteroid)

**Asteroide potencialmente peligroso.** Es la variable objetivo del proyecto. La
definición oficial (IAU / CNEOS) es una regla determinista de dos umbrales:

> Un objeto es PHA si **MOID ≤ 0.05 au** *y* **H ≤ 22 mag**.

Es decir: pasa lo bastante cerca *y* es lo bastante grande (≳140 m) como para causar
daño regional si impactara. Nótese que "potencialmente peligroso" **no** significa "en
curso de colisión": es una categoría de vigilancia, no una predicción de impacto.

En el proyecto hay **dos** versiones de la etiqueta:

| Etiqueta | Cómo se construye | Rol |
|---|---|---|
| `PHA_official` | Flag `pha` de la SBDB de JPL | *Ground truth* |
| `PHA_proxy` | `H` observado ≤ 22 **y** distancia mínima observada ≤ 0.05 au | Aproximación desde lo observable |

`PHA_proxy` sustituye el MOID (orbital, no observado) por la mínima distancia de
aproximación efectivamente registrada. Comparar ambas mide **cuánto se pierde** al
usar solo lo observado: en el dataset actual la correlación entre distancia mínima
observada y MOID es ≈ **0.869**.

---

## MOID (Minimum Orbit Intersection Distance)

**Distancia mínima de intersección orbital**: la menor distancia posible entre la órbita
del objeto y la órbita de la Tierra, consideradas como dos curvas geométricas en el
espacio.

La distinción crucial frente a la distancia de una aproximación observada:

- El **MOID** es una propiedad de la *geometría orbital*. Responde a "¿cuánto podrían
  llegar a acercarse las dos órbitas?" — independientemente de dónde estén los cuerpos
  en cada momento.
- La **distancia de aproximación** (`dist`, `dist_min`) es lo que realmente pasó en un
  encuentro concreto: depende de que ambos cuerpos coincidieran en fase.

La distancia observada es siempre **≥** el MOID (nunca puedes acercarte más de lo que la
geometría permite), y para objetos con pocos pasos registrados puede ser muchísimo mayor.
De ahí que `PHA_proxy` no reproduzca exactamente a `PHA_official`.

> ⚠ Esa desigualdad vale **a época fija**. El MOID evoluciona secularmente en escalas de
> siglos, así que comparar el MOID actual de la SBDB con pasos de 1900–2026 la rompe: en
> el snapshot actual, el **6.0 % de los objetos** tiene `MOID > distancia mínima
> observada`. Ver [discrepancia G](05-discrepancias.md#g--el-moid-depende-de-la-época-y-el-104--de-los-objetos-lo-demuestra).

Esta diferencia entre "lo que la geometría permite" y "lo que de hecho se observó" es el
**objeto de estudio** del trabajo. Entre los PHA reales, la distancia observada mínima es
mediana **3.8 veces mayor** que su MOID: rara vez se captura un objeto en la configuración
orbital más desfavorable, y de ahí que el proxy observacional tenga alta precisión pero
bajo *recall*.

**Por qué importa metodológicamente:** el MOID *define* la etiqueta. Usarlo como
predictor sería circular — el modelo estaría leyendo la respuesta. En este proyecto
`MOID (au)` se descarga y se guarda, pero **solo para etiquetar y validar, nunca como
feature**.

---

## Magnitud absoluta H

Brillo intrínseco del objeto, definido como la magnitud aparente que tendría si se
observara a **1 au del Sol y 1 au del observador, con ángulo de fase 0°**. Al fijar las
distancias, elimina el efecto de "está lejos, se ve poco" y deja una medida comparable
entre objetos.

Dos propiedades contraintuitivas de la escala de magnitudes:

- Es **inversa**: cuanto *menor* es H, *más brillante* — y, a albedo fijo, **más grande**
  es el objeto. `H ≤ 22` es el umbral de "grande" (≳140 m).
- Es **logarítmica**: 5 magnitudes = factor 100 en flujo luminoso.

Por eso, al agregar a nivel de objeto, la feature es `H_obs = min(H)`: el **mínimo** de H
corresponde a la estimación de mayor tamaño observada.

---

## Albedo y estimación del diámetro

El **albedo** (`p_V`) es la fracción de luz incidente que refleja la superficie. Un
objeto oscuro (carbonáceo, `p_V ≈ 0.05`) y uno claro (silicáceo, `p_V ≈ 0.25`) con el
mismo brillo tienen tamaños muy distintos.

Cuando la CAD API no reporta diámetro medido, el notebook de datos lo **imputa** desde H
con la relación estándar:

```
D(km) = (1329 / √p_V) · 10^(−0.2·H)     con p_V = 0.14 (albedo típico de NEO)
```

En el código esto aparece como:

```python
albedo = 1329 / math.sqrt(0.14)   # ojo: la variable NO guarda el albedo,
                                  # guarda el factor 1329/√p_V ≈ 3552
df.loc[sin_diametro, "diameter"] = albedo * (10 ** (-0.2 * df.loc[sin_diametro, "h"]))
```

> **Consecuencia metodológica documentada:** para todas esas filas, `Diameter(km)` es una
> función determinista de `H(mag)`. No son variables independientes. Por eso el análisis
> de colinealidad descarta `diam_max` y conserva `H_obs` como único indicador de tamaño.

La incertidumbre del diámetro imputado se fija en un 35 % (`Std Diameter(km) = 0.35·D`).

---

## Unidad astronómica (au)

Distancia media Tierra–Sol: **≈ 149 597 871 km**. Todas las distancias del dataset están
en au.

Referencias útiles para leer las cifras:

| Distancia | En au | Equivalente |
|---|---|---|
| Tierra–Luna | 0.00257 | 384 400 km |
| Umbral PHA | **0.05** | ≈ 7.5 millones de km ≈ 19.5 distancias lunares |
| Tierra–Sol | 1.0 | 149.6 millones de km |

---

## Velocidad relativa (`v_rel`) vs velocidad infinita (`v_inf`)

Ambas en km/s, ambas describen la rapidez del encuentro, pero en momentos distintos:

- **`v_rel`** — velocidad relativa al cuerpo central **en el instante de máxima
  aproximación**. Ya incluye la aceleración gravitatoria de la Tierra sobre el objeto
  (*focalización gravitatoria*).
- **`v_inf`** — velocidad relativa "en el infinito": la que tendría el objeto lejos del
  pozo de potencial terrestre, es decir, descontando esa aceleración.

Siempre `v_rel ≥ v_inf`. Para pasos lejanos la diferencia es despreciable y ambas son
casi idénticas — de ahí que el análisis de colinealidad las encuentre redundantes y
conserve una sola (`vrel_max`).

---

## Arco orbital (`data_arc`) y fecha de primera observación (`first_obs`)

Metadatos de **calidad de la órbita**, no propiedades físicas del objeto:

- **`data_arc`** — número de días entre la primera y la última observación astrométrica
  del objeto. Un arco largo significa una órbita bien determinada; un arco de días
  significa una órbita provisional con incertidumbre grande.
- **`first_obs`** — fecha real del primer avistamiento (de la que se deriva
  `first_obs_year`). **No** confundir con el año del primer evento de aproximación del
  catálogo: la CAD API incluye aproximaciones *calculadas retroactivamente* hasta 1900
  para objetos descubiertos en 2020.

Estas dos columnas son **metadatos de selección**, no features. Sirven para el análisis
de la función de selección observacional: la prevalencia de PHA entre los objetos
descubiertos antes de 2015 es del 14.6 %, y entre los descubiertos después, del 2.6 % —
no porque el cielo haya cambiado, sino porque los sondeos antiguos solo veían los
objetos grandes y cercanos. `corr(data_arc, PHA) = 0.351` cuantifica el confundidor:
los PHA se caracterizan mejor precisamente por ser interesantes.

---

[← Índice](README.md) · [Siguiente: columnas del dataset →](02-columnas-del-dataset.md)
