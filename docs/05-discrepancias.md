# 05 — Discrepancias físicas y teóricas

[← Índice](README.md)

Auditoría del proyecto contra (a) la documentación oficial de las APIs de JPL, (b) la
literatura astronómica sobre NEOs y (c) otros trabajos de clasificación PHA con machine
learning. Todo lo que se afirma aquí es **verificable ejecutando**:

```bash
python scripts/verificar_discrepancias.py
```

Ese script imprime la evidencia numérica de cada apartado. Las secciones A–I describen el
problema **tal como se encontró** (con los números del snapshot censurado original,
32 568 eventos / 18 934 objetos, citados como evidencia forense) y, cuando aplica, un
bloque **"Resultado tras la corrección"** con los números reales del snapshot corregido
(340 469 eventos / 37 526 objetos con evento observado; `data/snapshot_info.json`).

> **Resumen ejecutivo.** La *tesis* del proyecto — que la literatura clasifica PHA de
> forma circular — queda **confirmada y reforzada** por la revisión bibliográfica. Pero
> la *implementación* reproducía una versión más sutil del mismo problema, y un parámetro
> de descarga no especificado censuraba el dataset exactamente en el umbral que define la
> etiqueta.

## Estado tras la corrección

Las discrepancias corregibles **están corregidas** y el control de integridad
(`python scripts/verificar_discrepancias.py`) pasa entero. Las heredadas de la definición
oficial de PHA o de los datos de JPL no son corregibles y quedan declaradas como
limitaciones.

| # | Discrepancia | Severidad | Estado |
|---|---|---|---|
| [A](#a--el-dataset-está-censurado-en-el-umbral-que-define-la-etiqueta) | Dataset censurado en 0.05 au (default de la CAD API) | **Crítica** | ✅ **Corregida** — `dist-max=0.5` explícito |
| [B](#b--hmag-de-la-cad-api-es-literalmente-la-variable-que-define-la-etiqueta) | `H(mag)` "observado" ≡ `H_SBDB(mag)` que define el target | **Crítica** | ✅ **Resuelta por reencuadre** — el trabajo pasa a ser sustitución de medida, no predicción |
| [C](#c--el-diámetro-no-es-una-variable-el-979--es-h-reescalado) | Features derivadas (`Diameter`, `v_rel`) degeneran el PCA | Alta | ✅ **Corregida** — conjunto reducido a 3 cantidades independientes |
| [D](#d--la-conclusión-sobre-el-sesgo-de-muestreo-es-incorrecta) | Conclusión sobre el sesgo de muestreo incorrecta | Alta | ✅ **Corregida** — se reporta por condición; el sesgo además casi desaparece |
| [E](#e--dist_min-es-una-cota-3-sigma-no-una-distancia-física) | `dist_min` es una cota 3σ, no una distancia | Media | ✅ **Corregida** — se usa la nominal; la incertidumbre entra aparte |
| [F](#f--el-436--de-las-aproximaciones-observadas-son-anteriores-al-descubrimiento) | Aproximaciones anteriores al descubrimiento | Media | ✅ **Corregida** — flag `post_discovery` + filtro y anexo |
| [G](#g--el-moid-depende-de-la-época-y-el-104--de-los-objetos-lo-demuestra) | El MOID es dependiente de época | Media | ⚠ **Heredada** — declarada; bajó del 10.4 % al 6.0 % |
| [H](#h--albedo-único-014-frente-a-una-distribución-bimodal) | Albedo único 0.14 frente a distribución bimodal | Media | ⚠ **Heredada** — la definición oficial es albedo-ciega |
| [I](#i--el-flag-oficial-no-es-100--reproducible) | El flag `pha` no es 100 % reproducible | Baja | ⚠ **Heredada** — techo de exactitud declarado |

### Efecto de la corrección sobre el dataset

| Métrica | Censurado | Corregido | Población NEO (SBDB) |
|---|---:|---:|---:|
| Eventos | 32 568 | 340 469 (74 530 observados) | — |
| Objetos | 18 934 | 37 526 | 42 318 |
| `max(dist)` | 0.0500 au | 0.5000 au | — |
| MOID ≤ 0.05 au | 99.7 % | **59.5 %** | 54.1 % |
| H ≤ 22 | 7.4 % | **21.3 %** | 27.6 % |
| Prevalencia PHA | 7.3 % | **6.0 %** | 6.1 % |
| `PHA ≡ (H ≤ 22)` | 99.6 % | **84.5 %** | — |
| `PHA_proxy ≡ (H ≤ 22)` | 100.0 % | **80.5 %** | — |

El punto importante: la prevalencia PHA del catálogo pasa de estar 1.2 pp por encima de
la poblacional a estar 0.1 pp por debajo, y la desviación en la condición MOID cae de
45.6 a **5.3 puntos porcentuales**. El catálogo deja de ser un muestreador sesgado.

---

## A — El dataset está censurado en el umbral que define la etiqueta

### Qué asume el proyecto

Que la consulta a la CAD API con `date-min=1900-01-01&date-max=hoy&diameter=true`
devuelve **todas** las aproximaciones cercanas registradas, y que por tanto
`CA DistanceMinimum (au)` tiene un rango informativo que permite discriminar objetos.

### Qué ocurre en realidad

La [documentación de la CAD API](https://ssd-api.jpl.nasa.gov/doc/cad.html) especifica
que el parámetro **`dist-max` tiene valor por defecto `0.05` (au)**. El notebook nunca lo
sobrescribe. Consecuencia directa:

```
max(CA DistanceNominal) = 0.050000 au        ← exactamente el umbral PHA
objetos con distmin_min <= 0.05 : 18 934 / 18 934   (100 %)
```

No es una coincidencia física: es el filtro de la consulta.

### Por qué es grave — tres consecuencias en cadena

**1. `PHA_proxy` no es lo que dice ser.** Su definición es
`(H_obs ≤ 22) ∧ (distmin_min ≤ 0.05)`. Como la segunda condición se cumple para el
100 % de los objetos, es **vacua**:

```
PHA_proxy == (H_obs <= 22)?   coincidencia = 100.0000 %
```

La etiqueta proxy no aproxima el MOID por la distancia observada. Es un umbral sobre H y
nada más. El sub-resultado "¿aproxima la distancia observada al MOID?" mide algo distinto
de lo que se cree medir: `corr(dist_min, MOID) = 0.869` es una correlación calculada
**dentro** de la región ya truncada, no en el rango completo.

**2. La condición MOID de la definición oficial también se satura.** Geométricamente, la
distancia real de un encuentro nunca puede ser menor que el MOID (`dist ≥ MOID`). Si el
catálogo solo contiene eventos con `dist ≤ 0.05 au`, entonces **todo objeto del catálogo
tiene MOID ≤ 0.05 au** por construcción:

```
objetos con MOID <= 0.05 au : 18 871 / 18 934   (99.7 %)
```

**3. Por tanto el target colapsa a un umbral sobre una sola variable:**

```
PHA_official == (H_sbdb <= 22)?   coincidencia = 99.641 %
```

### Impacto sobre los resultados del notebook

Esto explica —y desactiva— el hallazgo central. La tabla de features dice:

| Conjunto | F2 |
|---|---|
| `kin+size` | 0.977 |
| `kin-only` | 0.492 |
| `size-only` | 0.975 |

La lectura del notebook es *"la señal de peligrosidad la aporta el tamaño, no la
cinemática"*. Presentado como hallazgo físico, no lo es: en una muestra censurada en
`dist ≤ 0.05`, la mitad "distancia" de la definición PHA ya está satisfecha por todos,
así que lo único que queda por decidir es `H ≤ 22`. **La dimensión cinemática fue
colapsada por el filtro de la consulta antes de que ningún modelo viera los datos.**

Igualmente, el "baseline de regla de dos umbrales" que alcanza F2 = 0.980 es en realidad
una regla de **un** umbral.

### Corrección aplicada

`dist-max=0.5` explícito en la consulta CAD (`data/ProyectoNeoRework_data.ipynb`), más un
descargador por tramos de 10 años con reintentos — la petición única de ~340k eventos
falla de forma intermitente (`ChunkedEncodingError`) y no era reproducible.

### Resultado tras la corrección

```
objetos con distmin_min <= 0.05  : 46.6 %   (antes: 100 %)
objetos con MOID <= 0.05 au      : 59.5 %   (antes: 99.7 %)
PHA_official == (H_sbdb<=22)     : 84.5 %   (antes: 99.6 %)
```

Y el efecto en la clasificación es el que se predecía: la censura **inflaba
artificialmente** la regla de umbrales hasta igualar al ML. Corregida, la brecha es
grande y honesta:

| Conjunto | F2 (censurado) | F2 (corregido) |
|---|---:|---:|
| Regla de umbrales (baseline) | 0.980 | **0.343** |
| `kin+size` (XGBoost) | 0.977 | **0.712** |
| `kin-only` (XGBoost) | 0.492 | **0.381** |
| `size-only` (XGBoost) | 0.975 | **0.653** |

Ya no hay una tautología: hay un modelo (F2=0.712) claramente por encima de la regla
ingenua (F2=0.343), y `kin+size` supera a `size-only`, así que la cinemática **sí** aporta
señal por encima del tamaño solo — lo contrario de lo que sugería el dataset censurado.

---

## B — `H(mag)` de la CAD API es literalmente la variable que define la etiqueta

### Qué afirma el proyecto

El README y el notebook distinguen entre el **`H` observado** (columna `H(mag)`,
proveniente de la CAD API, usada como feature) y el **`H` oficial** (columna
`H_SBDB(mag)`, usada "solo para etiquetar y validar"). El diseño metodológico depende de
que sean cosas distintas.

### Qué ocurre en realidad

No son cosas distintas. Son el mismo número:

```
objetos con más de un valor distinto de H entre sus eventos : 0
correlación H(mag) vs H_SBDB(mag)                          : 1.000000
max |diferencia|                                           : 0.0050 mag   (redondeo)
```

La CAD API no mide H en cada aproximación: adjunta a cada evento el H **de catálogo** del
objeto, que es el mismo que sirve la SBDB. Por eso `min(H)` sobre los eventos de un objeto
es simplemente su H, y por eso `H_obs` y `H_sbdb` son intercambiables.

### Por qué es una discrepancia teórica

La pregunta de investigación es *"¿se puede inferir PHA sin las variables que lo
definen?"*. PHA se define por **dos** variables: MOID y H. El proyecto excluye
correctamente el MOID… y luego mete H en los conjuntos `kin+size` y `size-only`.

Es decir: **se está usando la mitad de la definición como predictor**. Combinado con la
discrepancia A (que hace vacua la otra mitad), el resultado es que `size-only` no predice
la etiqueta — la *calcula*.

Esto no es un descuido invisible: el notebook titula una sección "Circularidad /
*leakage*" y demuestra que una regla de umbrales iguala al ML. El problema es de
**framing**: se presenta la circularidad como un hallazgo *sobre la literatura*, cuando
también aplica al propio experimento. La sección de conclusiones debería decirlo.

### Corrección aplicada

Opción 3: **reencuadre de la pregunta**, ya no como predicción de PHA sino como
**sustitución de medida** — cuánto se pierde al reemplazar el MOID orbital por la
distancia observada, y cómo lo distorsiona la selección. El aprendizaje supervisado pasa
a usarse como instrumento de medida (para cuantificar cuánto aporta cada bloque de
variables), no como la contribución del trabajo. H se mantiene como feature porque excluirla
del todo (opción 1) desperdiciaría la comparación `kin-only` vs `size-only` vs `kin+size`,
que es en sí misma informativa.

### Resultado tras la corrección

Con el dataset corregido, el incremento de `kin+size` sobre `size-only` **ya no es nulo**:

```
size-only  (XGBoost) F2 = 0.656
kin+size   (XGBoost) F2 = 0.714      incremento = +0.058
```

La afirmación anterior de este documento ("ese incremento es ≈ 0.002, es decir, ninguno")
era un artefacto de la censura (discrepancia A) y queda revertida: para algoritmos como
XGBoost, la cinemática sí añade información medible (+0.058) por encima del tamaño solo, 
aunque el tamaño siga siendo el predictor individual más fuerte. (Cabe destacar que 
Random Forest no logra integrar bien la cinemática y sufre una caída de rendimiento, 
lo que subraya la ventaja del boosting para esta tarea).

---

## C — El diámetro no es una variable: el 97.9 % es H reescalado

### El conjunto "canónico de 5 features" tiene 3 dimensiones reales

Las cinco features del análisis no supervisado son
`dist`, `v_rel`, `v_inf`, `H`, `Diameter`. **Dos de ellas son funciones deterministas de
las otras**, y ambas relaciones se verifican numéricamente en el dataset.

**1. `Diameter` = f(`H`)** — el notebook de datos imputa
`D = (1329/√0.14)·10^(−0.2H)` cuando falta el diámetro, y falta casi siempre:

```
filas con diámetro == fórmula desde H : 31 879 / 32 560   (97.9 %)
filas con diámetro realmente medido   :      681           ( 2.1 %)
```

**2. `v_rel` = f(`v_inf`, `dist`)** — esto no está documentado en el proyecto, que trata
la redundancia de velocidades como correlación empírica (`|corr| > 0.9`). En realidad
JPL deriva `v_rel` de `v_inf` con la ecuación de energía de un encuentro hiperbólico,
`v_rel² = v_inf² + 2μ⊕/r`, y la identidad se reproduce **a precisión de máquina**:

```
v_rel − v_inf          : mediana 0.01279 km/s   max 11.17959 km/s
predicción 2-cuerpos   : mediana 0.01279 km/s   max 11.17959 km/s
error mediano |obs−teo|: 0.00000 km/s
violaciones v_rel < v_inf : 0 / 32 550
```

(Nótese que la diferencia **no** es despreciable en todo el rango: alcanza 11 km/s en los
pasos más rasantes. Lo que la hace redundante no es su pequeñez, sino que `dist` ya está
en el conjunto de features.)

Quedan por tanto **tres cantidades independientes**: `dist`, `v_inf` y `H`.

### Consecuencias

- El resultado de PCA **"PC5 ≈ 0 % de varianza"**, presentado en el README como evidencia
  de redundancia física entre las variables de los NEOs, es un **artefacto aritmético**.
  Dos relaciones funcionales exactas producen necesariamente autovalores degenerados. No
  dice nada sobre asteroides: dice que se metieron dos columnas derivadas.
- Lo mismo afecta a `PC1+PC2 ≈ 77.8 %`: la varianza total está inflada por dos
  dimensiones duplicadas, así que el porcentaje no es comparable con el de un análisis
  sobre variables independientes.
- Los clusters de K-Means se calculan en un espacio donde dos de los cinco ejes son
  redundantes, lo que **pondera doblemente** el tamaño (H y Diameter) y la velocidad
  (v_rel y v_inf) frente a la distancia. Las interpretaciones de los clusters
  ("pequeños–medianos y rápidos") heredan ese sesgo de ponderación.

### Corrección aplicada

PCA/K-Means ahora corren sobre `["CA DistanceNominal (au)", "V infinity(km/s)",
"H(mag)"]` — las tres cantidades independientes. (t-SNE se eliminó del notebook por
completo: era puramente ilustrativo, no sustentaba ninguna conclusión, y su costo
computacional —incluso en la versión reducida a un solo embedding— no se justificaba.)

### Resultado tras la corrección

```
PC1: 53.3%   PC2: 24.8%   PC3: 21.9%      PC1+PC2 = 78.1 %
```

Con solo 3 features, ya no hay una cuarta/quinta componente degenerada por aritmética: las
tres capturan varianza real. El umbral de suficiencia (80 %) sigue sin alcanzarse por muy
poco, así que la justificación de clusterizar sobre el espacio completo (en vez de sobre
PC1+PC2) se mantiene, pero ahora por una razón física y no por un artefacto de columnas
duplicadas.

---

## D — La conclusión sobre el sesgo de muestreo es incorrecta

### Qué concluye el notebook

> "Comparado con el total de NEOs (SBDB), la prevalencia PHA está solo levemente
> enriquecida — la selección dominante es temporal (descubrimiento), no el
> condicionamiento por aproximación."

### Qué muestran los datos

| población | n | PHA | MOID ≤ 0.05 | H ≤ 22 |
|---|---:|---:|---:|---:|
| todos los NEOs (SBDB) | 42 074 | 6.1 % | **54.1 %** | **27.7 %** |
| en el catálogo CAD | 18 934 | 7.3 % | **99.7 %** | **7.4 %** |
| NEOs no capturados | 23 140 | 5.1 % | 16.5 % | 44.3 % |

El condicionamiento por aproximación cercana **no** es leve. Es enorme, y actúa en las
dos dimensiones de la definición PHA en **direcciones opuestas**:

- **Satura el MOID**: 54.1 % → 99.7 %. Todo objeto del catálogo cumple ya media
  definición de PHA.
- **Vacía el tamaño**: 27.7 % → 7.4 %. El catálogo está fuertemente **empobrecido** en
  objetos grandes, porque los objetos pequeños solo entran en cualquier catálogo cuando
  pasan cerca, mientras que los grandes se descubren estén donde estén.

La prevalencia PHA apenas se mueve (6.1 % → 7.3 %) porque es el **producto** de las dos
condiciones y los dos sesgos casi se cancelan. Inferir "sesgo leve" a partir de la
prevalencia conjunta es exactamente el error que el propio proyecto denuncia en otros
contextos: mirar el agregado y no los factores.

### Corrección aplicada

El notebook reporta ahora el sesgo **marginal por condición**, no solo la prevalencia.

### Resultado tras la corrección

```
población               n       PHA    MOID<=0.05   H<=22
todos los NEOs (SBDB)   42 318  6.1 %  54.1 %       27.6 %
en el catálogo CAD      37 405  6.0 %  59.4 %       21.4 %
NEOs no capturados       4 913  6.2 %  12.5 %       75.2 %
```

Al descensurar el catálogo (discrepancia A), el sesgo de muestreo **casi desaparece**: la
desviación en la condición MOID cae de 45.6 a **5.3 puntos porcentuales**, y la
prevalencia PHA queda prácticamente igual a la poblacional (6.0 % vs 6.1 %). El catálogo
deja de ser un muestreador fuertemente sesgado por aproximación — el efecto que quedaba
era, en su mayor parte, la censura de la consulta, no un sesgo observacional intrínseco.

---

## E — `dist_min` es una cota 3-sigma, no una distancia física

La [documentación de la CAD API](https://ssd-api.jpl.nasa.gov/doc/cad.html) define
`dist_min` como *"minimum (**3-sigma**) approach distance"*. Es el borde inferior del
intervalo de confianza que induce la **incertidumbre orbital**, no una distancia medida.

El proyecto lo usa como aproximación del MOID y como feature principal (`distmin_min`),
agregando además con `min` sobre todos los eventos del objeto. Eso encadena tres
problemas:

1. Se mezcla **proximidad física** con **mala determinación orbital**. Dos objetos que
   pasaron a la misma distancia tienen `dist_min` muy distintos si uno tiene arco de
   30 años y el otro de 3 días.
2. El `min` sobre eventos es un **estadístico de orden extremo** sobre una cota inferior:
   selecciona sistemáticamente el evento con mayor incertidumbre.
3. Esa incertidumbre depende de `data_arc`, que es **precisamente el confundidor de
   selección** que el notebook estudia (`corr(data_arc, PHA) = 0.351`). La feature y el
   confundidor comparten fuente.

Evidencia de que la cota no es física:

```
objetos con min(dist_min) == 0.0 exactamente  : 10
objetos con min(dist_min) < 1e-5 au (<1500 km): 26
```

Un `dist_min` de 0.0 au no significa un impacto: significa que la incertidumbre 3σ del
paso abarca la posición de la Tierra.

### Corrección aplicada

El conjunto de features usa `distnom_min` (la nominal); `distmin_min` se conserva solo
para evaluar el proxy contra el MOID (discrepancia central del trabajo), y
`dist_unc_med = mediana(dist − dist_min)` entra aparte como variable de calidad orbital.

---

## F — El 43.6 % de las aproximaciones "observadas" son anteriores al descubrimiento

```
eventos anteriores al descubrimiento : 14 198 / 32 568   (43.6 %)
objetos afectados                    :  7 473 / 18 934
mediana de años de retroacción       : 68
```

Con `date-min=1900-01-01`, la CAD API devuelve aproximaciones **calculadas por
integración numérica hacia atrás** para objetos descubiertos décadas después. Casi la
mitad del dataset no son observaciones.

Dos consecuencias:

1. **El framing "solo lo observado" es inexacto.** Esos eventos son salidas de un modelo
   dinámico que se alimenta de los elementos orbitales — las mismas variables que el
   proyecto se propone no usar. La exclusión del MOID es real; la exclusión de la
   información orbital, solo parcial.
2. **`n_appro` no mide exposición observacional.** Mide la ventana de integración
   (1900 → hoy) modulada por el periodo orbital y por la calidad de la órbita. Un objeto
   con órbita bien determinada admite integración retroactiva fiable y acumula eventos;
   uno con arco corto, no. `n_appro` es un proxy de `data_arc` disfrazado de feature
   física — y el notebook lo trata como "exposición observacional".

### Corrección aplicada

Columna `post_discovery` (1 = evento posterior al descubrimiento del objeto). El análisis
principal filtra a `post_discovery == 1`; el catálogo completo queda disponible para un
anexo de sensibilidad.

### Resultado tras la corrección

Con el catálogo descensurado (10.4× más eventos), la proporción de eventos retroactivos
sube al **78.1 %** (74 530 observados de 340 469): al ampliar la ventana de distancia, la
mayoría de los eventos nuevos son integraciones lejanas, no observaciones nuevas. El
filtro por `post_discovery` es, por tanto, más importante que antes, no menos.

También se midió el coste de **no** filtrar: incluir los eventos retroactivos infla la
calidad aparente del proxy MOID — `corr(dist_min, MOID)` sube de 0.868 (solo observado) a
0.931 (catálogo completo) — precisamente porque los eventos integrados numéricamente
tienden a acercarse más al mínimo orbital teórico que una observación real esporádica.

---

## G — El MOID depende de la época, y el 10.4 % de los objetos lo demuestra

```
objetos con MOID > min(distancia nominal observada) : 1 971 / 18 934   (10.4 %)
```

A época fija esto es **geométricamente imposible**: la distancia de un encuentro real no
puede ser menor que la mínima distancia entre las órbitas. La explicación es que se están
comparando cosas de épocas distintas — un MOID de la solución orbital **actual** contra
pasos repartidos entre 1900 y 2026.

El MOID evoluciona secularmente por la precesión del argumento del perihelio y del nodo
inducida sobre todo por Júpiter, con ciclos de Lidov–Kozai en escalas de miles de años;
la literatura documenta objetos que **entran y salen** de la condición `MOID ≤ 0.05 au`
en escalas de siglos ([Classifying and Characterizing the Evolution of the MOID for
NEAs](https://iopscience.iop.org/article/10.3847/PSJ/add323), PSJ 2025).

Es decir: **`PHA_official` no es una etiqueta estática** sobre un intervalo de 126 años,
pero el proyecto la trata como tal. Esto contamina especialmente el hold-out temporal,
donde el eje de análisis es precisamente el tiempo.

### Limitación declarada (no corregible)

No se puede "arreglar" — es inherente a comparar un MOID de época actual contra un
catálogo de 126 años. Se documenta explícitamente en vez de ignorarla. Con el catálogo
corregido y restringido a lo observado, la incidencia **baja** de 10.4 % a **6.0 %**
(probablemente porque los eventos retroactivos de la discrepancia F, al acercarse más al
MOID teórico, eran más propensos a violarlo por errores de redondeo de época), pero sigue
sin ser cero y no se puede reducir más sin remodelar dinámicamente cada órbita.

---

## H — Albedo único 0.14 frente a una distribución bimodal

La imputación usa `p_V = 0.14` para todos los objetos. La relación
`D_km = 1329·10^(−H/5)/√p_V` y el valor 0.14 son los estándar
([ESA NEOCC](https://neo.ssa.esa.int/definitions-assumptions)), y en efecto reproducen
`H = 22 ↔ 140 m` — la equivalencia que cita el proyecto es correcta.

El problema es que el albedo de los NEA **no es unimodal**. Wright et al. (2016), con
428 NEAs de WISE, encuentran dos poblaciones: un 25.3 % oscuro con pico en
`p_V ≈ 0.030` y un 74.7 % moderado con pico en `p_V ≈ 0.168`
([arXiv:1606.07421](https://arxiv.org/abs/1606.07421)).

```
p_V=0.030  población oscura   D_real/D_asumido = 2.16x
p_V=0.168  población clara    D_real/D_asumido = 0.91x

Umbral de H equivalente a 140 m:
  p_V=0.030 -> H = 23.69      p_V=0.140 -> H = 22.02      p_V=0.168 -> H = 21.82
```

Para una cuarta parte de los NEAs, el diámetro imputado subestima el real en un factor
**2.2**, y el corte "140 m" equivaldría a `H ≤ 23.7` en vez de `H ≤ 22.0`.

**Esta discrepancia es heredada, no introducida:** la definición oficial de PHA es
deliberadamente albedo-ciega, porque el albedo solo se conoce para una minoría de
objetos. Usar 0.14 es lo correcto para reproducir el criterio oficial. Lo que falta en el
proyecto es **declarar** que el "diámetro" del dataset no es una estimación física del
tamaño sino una reparametrización de H bajo un albedo convencional, y que por tanto no
debe interpretarse físicamente en las conclusiones de los clusters
(p. ej. "Cluster 2 — pequeños–medianos").

---

## I — El flag oficial no es 100 % reproducible

```
regla exacta (H<=22 & MOID<=0.05) vs flag pha : 99.831 %
PHA=1 con H>22                : 23   (rango H 22.01–22.57)
H<=22 y MOID<=0.05 pero PHA=0 :  9
```

32 objetos de 18 927 no encajan en su propia definición usando los campos actuales de la
SBDB. Todas las excepciones son **marginales** (H entre 22.01 y 22.57), lo que apunta a
la causa: el flag se asignó con los valores de H y MOID vigentes en el momento de la
designación, y ambos se revisan al mejorar la astrometría. No hay error en el proyecto;
conviene mencionarlo como cota superior de la exactitud alcanzable (≈ 99.8 %), porque un
modelo que reporte más que eso está sobreajustando ruido de catálogo.

---

## Contraste con la literatura y otros repositorios

### La tesis del proyecto se confirma

La revisión respalda de forma contundente la crítica que motiva el proyecto.

- **Graph Neural Networks (arXiv:2504.18605, 2025)** — usa como features "eccentricity,
  semimajor axis, perihelion distance, **absolute magnitude**, diameter, and the NEO and
  **PHA flags**", incluyendo MOID. No menciona en ningún momento que MOID y H *definan*
  la etiqueta. Reporta 99 % de accuracy y AUC 0.99 — pero para la clase peligrosa,
  precisión **24 %** y F1 **0.37**. Es el ejemplo perfecto de las dos patologías que el
  proyecto denuncia: circularidad + accuracy engañosa con clases desbalanceadas.
- **Estudios de importancia de variables** reportan que "MOID and Absolute Magnitude
  accounted for over 77 % of the importance of the features". Traducción: el 77 % del
  poder predictivo procede de las dos variables que constituyen la definición.
- **Repos del dataset Kaggle "NASA Nearest Earth Objects"** (asoderlund/NEO-Analysis,
  Ahmed9Elsayed/NEOs-classification, doguilmak/Nearest-Earth-Objects-Classification,
  entre otros) — usan `est_diameter_max` y `absolute_magnitude` **juntos**, siendo el
  primero una función determinista del segundo. Uno de ellos observa "a very clear
  correlation between est_diameter_max and absolute_magnitude" y decide **conservar
  ambos** porque "most metrics were improved". Reportan ~93 % de accuracy sobre una clase
  positiva del 9.7 %, es decir, apenas por encima del 90.3 % que da predecir siempre "no
  peligroso"; y ninguno define el criterio oficial de PHA.

Frente a esto, el proyecto hace tres cosas que la literatura revisada **no** hace, y son
sus aportaciones reales: excluir el MOID de las features, omitir deliberadamente la
accuracy en favor de F2/PR-AUC, y validar con un hold-out temporal por fecha real de
descubrimiento.

### Dónde el proyecto quedaba más débil que la literatura (ya corregido)

Los repos de Kaggle, con todos sus defectos, trabajan sobre distancias de encuentro sin
truncar (decenas de millones de km). La versión original de este dataset estaba censurada
en 7.5 millones de km (discrepancia A), lo que le impedía observar el régimen donde la
cinemática podría discriminar — y de hecho, corregida la censura, la cinemática **sí**
discrimina (`kin+size` supera a `size-only` en F2, algo que el dataset truncado no podía
mostrar). Sigue sin tratarse la evolución secular del MOID (G): ninguno de los trabajos
revisados lo hace tampoco, pero al extender el análisis a 126 años el proyecto se expone
algo más a ese efecto que un estudio de una sola época.

### Fuentes

- [JPL SBDB Close-Approach Data API — documentación](https://ssd-api.jpl.nasa.gov/doc/cad.html)
- [JPL SBDB Query API — documentación](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html)
- [ESA NEOCC — Definitions & Assumptions](https://neo.ssa.esa.int/definitions-assumptions)
- [Wright et al. (2016), *The Albedo Distribution of Near Earth Asteroids*](https://arxiv.org/abs/1606.07421)
- [*Classifying and Characterizing the Evolution of the MOID for NEAs*, PSJ (2025)](https://iopscience.iop.org/article/10.3847/PSJ/add323)
- [*Explainable Deep-Learning Based PHA Classification Using GNNs*, arXiv:2504.18605](https://arxiv.org/html/2504.18605v1)
- [*The Hazardous km-sized NEOs of the Next Thousands of Years*, AJ (2023)](https://iopscience.iop.org/article/10.3847/1538-3881/acd378)
- [asoderlund/NEO-Analysis](https://github.com/asoderlund/NEO-Analysis) · [Ahmed9Elsayed/NEOs-classification](https://github.com/Ahmed9Elsayed/NEOs-classification) · [doguilmak/Nearest-Earth-Objects-Classification](https://github.com/doguilmak/Nearest-Earth-Objects-Classification)

---

## Dónde **no** hay discrepancia, y por qué

Igual de importante: estas decisiones del proyecto se revisaron y **son correctas**. Se
documentan aquí para que nadie las "arregle" por error.

| Decisión del proyecto | Por qué es correcta |
|---|---|
| **Definición de PHA como `MOID ≤ 0.05 au ∧ H ≤ 22`** | Coincide exactamente con el criterio IAU/CNEOS y con la [documentación de ESA NEOCC](https://neo.ssa.esa.int/definitions-assumptions). La equivalencia `H = 22 ↔ 140 m` bajo `p_V = 0.14` es la estándar y se reproduce numéricamente (H = 22.02). |
| **Constante 1329 en la relación H–diámetro** | Es la constante estándar de `D_km = 1329·10^(−H/5)/√p_V`. Verificada contra la literatura. Correcta. |
| **`v_rel ≥ v_inf`, y la poda que conserva una sola velocidad** | Correcto, y **mejor justificado de lo que el notebook afirma**. No es "correlación empírica alta": es una identidad exacta, ver [discrepancia C](#c--el-diámetro-no-es-una-variable-el-979--es-h-reescalado). La desigualdad se cumple en los 32 550 eventos comparables sin una sola violación. |
| **Clustering en el espacio estandarizado y no sobre coordenadas PCA** | Metodológicamente correcto y explícitamente documentado. Agrupar sobre componentes reducidas descartaría información. (Ahora 3-D tras la discrepancia C; el principio es el mismo.) |
| **Omitir accuracy; usar F2 + PR-AUC** | Correcto y por encima del estándar de la literatura revisada. Con ~6 % de positivos, la accuracy es inútil y F2 refleja bien el coste asimétrico del dominio. La elección de β = 2 está justificada. |
| **Agregación a nivel de objeto antes de la CV** | Correcto y necesario. Hacer CV a nivel de evento pondría el mismo asteroide en train y test. Este error sí está presente en varios de los repos revisados. |
| **Hold-out temporal por `first_obs` y no por año del primer evento** | **Es la decisión más acertada del proyecto.** Dado el hallazgo F (78.1 % de eventos retroactivos en el catálogo descensurado), usar el año del primer evento habría sido catastrófico. |
| **Imputar las medianas del train, no globales, en el hold-out temporal** | Correcto: evita fuga del futuro hacia el pasado. Bien implementado. |
| **`random_state = 20` propagado desde una sola variable** | Buena práctica de reproducibilidad, correctamente implementada en los tres modelos y los dos esquemas de CV. |
| **Excluir `MOID (au)` de las features** | Correcto y es la aportación metodológica central frente a la literatura. La discrepancia B no invalida esta decisión: la hace insuficiente por sí sola, resuelta por el reencuadre a sustitución de medida. |
| **Optimizar el umbral de decisión por F2 en vez de usar 0.5** | Correcto. 0.5 es una convención sin fundamento; se calcula con probabilidades fuera de fold, que es la forma adecuada. |

---

## Estado final

Todas las discrepancias corregibles (A–F) están **corregidas y verificadas ejecutando el
pipeline completo**: `python scripts/verificar_discrepancias.py` pasa entero, y el
notebook de ML corre de principio a fin sobre el dataset descensurado sin errores. Las
heredadas (G, H, I) quedan documentadas como limitaciones de la definición oficial de PHA
o de los datos de origen, no del proyecto.

El efecto más importante para el paper: la conclusión "la peligrosidad es casi trivial y
casi circular" del dataset censurado **se revierte**. Con los datos corregidos, la
clasificación exige un modelo real (F2=0.712 vs baseline de 0.343) y la cinemática aporta
información medible por encima del tamaño (F2 +0.059 sobre `size-only`). El resultado
central del trabajo — la sustitución directa de las dos condiciones de la definición PHA
(distancia observada ≤ 0.05 y H ≤ 22) tiene alta precisión (0.985) pero bajo recall
(0.295, a diferencia del proxy simple sin H que tiene recall 0.785 pero aún falla), con
una brecha física de 3.8× entre la distancia observada y el MOID entre los PHA reales —
se mantiene y se fortalece: ya no compite con un dataset degenerado por censura.

---

[← Fuentes de datos](04-fuentes-de-datos.md) · [Índice](README.md)
