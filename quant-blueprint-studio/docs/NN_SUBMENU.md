# Submenú 17 · ML / Redes Neuronales · Time Series & yFinance

**Entorno destino:** Quant Blueprint Studio, el editor visual propio del proyecto (React Flow + motor Python con DAG de Kahn).
No es Unreal, ComfyUI ni Blender: los nodos se declaran en `src/renderer/types/blueprint.ts` (inspector), sus pines en
`src/renderer/data/node-contracts.json` y su lógica en `backend/engine/nodes/nn_blueprint_nodes.py`.

**Contrato común de todos los nodos `nn_*`:**

- Pin de entrada `frame` y pin de salida `frame` (naranja). `nn_data` no tiene entrada; `nn_finetune` añade el pin `pretrained`.
- El `frame` lleva un DataFrame de pandas y un identificador de estado (`frame.attrs['nn']`). Los tensores, escaladores y redes viven en
  un registro en proceso (`pipeline_state`), así que dos ramas que salen del mismo nodo no se pisan.
- Cada nodo publica un informe (tabla, métricas, gráficos y notas) debajo del lienzo.
- Cada nodo valida sus parámetros y exige que sus prerrequisitos estén aguas arriba. Si algo falla, lanza `ValueError` con un mensaje
  en español que dice qué hacer. El lienzo además marca los errores de orden antes de ejecutar (`validation.ts`).
- Todo lo que se guarda en disco va a `artifacts/nn/` (`cache/`, `scalers/`, `models/`, `exports/`). Se puede cambiar con la variable
  `QL_ARTIFACTS`.

---

## 1 · Mapa del submenú

| # | Nodo (kind) | Rol |
|---|---|---|
| 1 | Datos yfinance (NN) · `nn_data` | Ingesta |
| 2 | Guardia de errores · `nn_guard` | Transversal: se coloca donde haga falta |
| 3 | Capa de entrada · `nn_input` | Construye X / y y define la forma de entrada |
| 4 | TR / VA / TE · `nn_split` | Etapas sin fuga |
| 5 | Escalador · `nn_scaler` | Normalización ajustada sólo con TR |
| 6 | Embeddings · `nn_embedding` | Representaciones densas (opcional) |
| 7 | Capa oculta · `nn_hidden` | Duplicable y encadenable |
| 8 | Capa de salida · `nn_output` | Cabeza, pérdida y compilación |
| 9 | Entrenar · `nn_train` | Ajuste con checkpoint en VA |
| 10 | Ajuste fino · `nn_finetune` | Reutiliza un modelo base |
| 11 | Predicción e inferencia · `nn_predict` | TR/VA/TE, en vivo y exportación |

### Flujo típico (entrenar y predecir)

```
nn_data ─► nn_guard ─► nn_input ─► nn_split ─► nn_scaler ─► [nn_embedding] ─► nn_hidden(A) ─► nn_hidden(B) ─► … ─► nn_output ─► nn_train ─► nn_predict
```

### Variante · ajuste fino (dos ramas)

```
Rama base:  nn_data(SPY) ─► … ─► nn_output ─► nn_train(save_as) ──┐ pin pretrained
Rama nueva: nn_data(QQQ) ─► nn_guard ─► nn_input ─► nn_split ─► nn_scaler ─► [nn_embedding] ─► nn_finetune ─► nn_predict
```

Alternativa sin cable: en `nn_finetune` pon `source = file` y carga el `.pt` que guardó un `nn_train` en otra ejecución.

### Variante · sólo inferencia con un escalador guardado

```
nn_data ─► nn_input (misma configuración) ─► nn_split ─► nn_scaler(load_transform) ─► nn_finetune(source=file, epochs bajos) ─► nn_predict
```

La predicción en vivo sale siempre de las últimas barras: tienen rasgos, pero su objetivo todavía es futuro.

### Variante · validación walk-forward

En `nn_split` pon `mode = walk_forward`. Con eso `nn_train` entrena un modelo por fold con ventana expansiva, reporta la media en VA y
entrena el modelo final en TR.

### Orden que exige el motor

| Nodo | Necesita aguas arriba |
|---|---|
| `nn_split` | `nn_input` |
| `nn_scaler` | `nn_split` |
| `nn_embedding` | `nn_scaler` |
| `nn_hidden` | `nn_input` |
| `nn_output` | `nn_scaler` |
| `nn_train` | `nn_output` |
| `nn_finetune` | `nn_scaler` (más el pin `pretrained` o un archivo) |
| `nn_predict` | `nn_train` o `nn_finetune` |

---

## 2 · Herramientas

### 1. Datos yfinance (NN) · `nn_data`

- **Propósito:** descargar OHLCV de Yahoo de forma robusta.
- **Entradas:** ninguna.
- **Configuración:**
  - `source` (yfinance / synthetic).
  - `ticker`: uno o varios separados por coma.
  - `interval`, `range_mode`, `period` o `start`/`end`.
  - `price_column` (close / adjclose) y `auto_adjust`.
  - `retries`, `fallback` (none / cache / synthetic), `cache_seconds`.
  - `synthetic_rows` y `seed`.
- **Salidas:** `frame` con `Open, High, Low, Close, AdjClose, Volume, Dividends, StockSplits, Ticker`. El índice es `Date`, sin zona
  horaria y ordenado.
- **Lógica:**
  - Reutiliza el validador y la caché en memoria de `yfinance_data`. Aplana el multi-índice y convierte el intradía a la zona elegida
    antes de quitarla.
  - Reintenta los fallos de red con espera exponencial (0.5 s, 1 s, 2 s…). Un ticker inválido no se reintenta.
  - Si la descarga funciona, guarda la serie en `artifacts/nn/cache`.
  - Con varios tickers se queda con la intersección de fechas.
- **Errores / edge cases:**
  - Ticker sin datos o caída de la API: aplica el fallback y lo anota en la bitácora. Con `fallback = none` detiene la ejecución con un
    diagnóstico.
  - Calendarios desalineados con menos de 100 fechas comunes (por ejemplo, cripto mezclado con acciones): error.
  - Límites de Yahoo para intradía (máximo de días por descarga y antigüedad máxima): los rechaza el validador existente.

### 2. Guardia de errores · `nn_guard`

- **Propósito:** gestión de errores transversal. Diagnostica, repara y centraliza los avisos de todos los nodos `nn_*`.
- **Entradas:** `frame`. Se puede poner después de cualquier nodo del submenú.
- **Configuración:**
  - `policy`: `fail` detiene, `repair` corrige, `warn` sólo registra.
  - `max_nan_ratio`.
  - `jump_threshold` (salto máximo de |log r|).
- **Salidas:** el `frame`, reparado si la política es `repair`. El informe muestra la tabla de controles y la bitácora acumulada
  (reintentos, fallbacks, reparaciones y eventos de NaN en el entrenamiento).
- **Lógica:**
  - Controles: Inf, NaN, fechas duplicadas (por ticker), orden cronológico, precios ≤ 0 y saltos extremos.
  - Si la guardia va después de la capa de entrada, comprueba también que `X`/`Xs` sean finitos y que X e y tengan las mismas filas.
  - Las reparaciones son causales: Inf → NaN, `ffill` por ticker (nunca `bfill`), eliminar duplicados conservando el último y ordenar.
- **Errores / edge cases:**
  - Con `fail`, el error enumera todos los controles fallidos.
  - Una X con NaN después de la capa de entrada siempre detiene la ejecución, porque no se puede reparar sin fuga.

### 3. Capa de entrada · `nn_input`

- **Propósito:** pasar de un único parámetro a términos polinómicos o ventanas de lags, y definir la forma de entrada y el
  optimizador base.
- **Entradas:** `frame` de precios.
- **Configuración:**
  - `mode`: simple (exactamente 1 parámetro), polynomial o lags.
  - `features`: `LogReturn`, `Range` (log High/Low), `LogVolume` o cualquier columna numérica.
  - Para lags: `lags` (k) y `lag_stride`.
  - Para polinomios: `degree` (1–4) e `interaction_only`.
  - `nan_policy` para los lags incompletos del principio: drop, zero o ffill.
  - `target` y `horizon` h (objetivo en t+1 … t+h).
  - `projection_units`: una capa densa opcional; 0 = identidad.
  - `activation`: linear, relu, leaky_relu, gelu, tanh, sigmoid, swish/silu o elu.
  - Optimizador base: `optimizer` (adam / adamw / sgd / rmsprop), `lr`, `weight_decay`, `momentum` y `clip`.
- **Salidas:** en el estado deja `X (n, d)`, `y (n, h)`, los nombres de los rasgos, fechas, tickers, `X_live` (las filas para inferencia
  en vivo) e `input_spec` = `{mode, d, shape: [None, d], seq_shape: [None, k, f], activation, projection_units, optimizer}`.
- **Lógica:**
  - Los lags se calculan con `shift(lag)` por ticker, así que todo es conocido al cierre de t. El objetivo usa `shift(-j)`.
  - Los polinomios generan potencias y productos cruzados.
  - La ventana de lags se aplana para la MLP, y el informe muestra también su forma secuencial equivalente.
- **Errores / edge cases:**
  - El modo simple con más de un parámetro es un error.
  - Una columna inexistente da error con la lista de las disponibles.
  - Menos de 100 filas útiles: error que propone reducir k o h.
  - Más de 4096 términos polinómicos: error.
  - Una activación no lineal sin proyección genera un aviso.

### 4. TR / VA / TE · `nn_split`

- **Propósito:** separar las etapas sin fuga.
- **Entradas:** `frame` con el estado de `nn_input`.
- **Configuración:**
  - `train_ratio` y `val_ratio` (TR + VA < 0.95).
  - `gap`: hueco adicional entre bloques.
  - `mode`: holdout o walk_forward, con `folds`.
- **Salidas:** el vector de grupo (0 TR, 1 VA, 2 TE, −1 purgado), los cortes `VAL_START`/`TEST_START`, los folds de walk-forward y la
  columna `Split` en el `frame`.
- **Lógica:**
  - Los cortes van por fecha, así que con varios tickers caen en el mismo instante para todos. No se baraja nada.
  - Purga de `h + gap` filas antes de cada corte, para que ninguna etiqueta vea el bloque siguiente.
  - Walk-forward usa ventana expansiva dentro de TR+VA y respeta la purga.
- **Errores / edge cases:** cada bloque necesita al menos 30 filas, y TR + VA debe ser < 0.95. El lienzo también marca esto último
  antes de ejecutar.

### 5. Escalador · `nn_scaler`

- **Propósito:** normalizar rasgos y objetivo sin fuga.
- **Entradas:** `frame` con el estado de `nn_split`.
- **Configuración:**
  - `operation`: `fit_transform` (entrenamiento) o `load_transform` (inferencia o ajuste fino con un escalador guardado).
  - `method`: standard, minmax, robust, maxabs o none (pass-through).
  - `feature_range` para MinMax: 0,1 o −1,1.
  - `scale_target`, `persist` y `save_as`.
- **Salidas:** `Xs`, `ys` y `Xs_live` en float32, junto con los objetos `x_scaler`/`y_scaler`. Con `persist`, el archivo
  `artifacts/nn/scalers/<save_as>.joblib` = `{x, y, method, names}`.
- **Lógica:** ajusta sólo con TR y transforma TR/VA/TE/en vivo. `nn_predict` aplica `inverse_transform` para volver a unidades reales.
- **Errores / edge cases:**
  - Archivo inexistente: error.
  - Rasgos distintos a los del escalador guardado: error, porque la capa de entrada debe coincidir.
  - Rasgos constantes en TR: aviso en la bitácora.

### 6. Embeddings · `nn_embedding`

- **Propósito:** representaciones densas aprendidas para el día de la semana, el mes (estacionalidad), el régimen de volatilidad y el
  activo.
- **Entradas:** `frame` con el estado de `nn_scaler`.
- **Configuración:** `weekday`, `month`, `regime`, `ticker` (sólo con varios tickers), `dim` (1–64) y `fusion`.
- **Salidas:** índices categóricos `cat (n, m)`, `cat_live` y `spec {names, cards, dim, fusion}`.
- **Lógica:**
  - Un `nn.Embedding(cards, dim)` por campo, que se aprende junto con la red.
  - `concat` une `[x, e₁…eₘ]` a la entrada. `add` proyecta cada embedding a d y lo suma a x.
  - El régimen son terciles de volatilidad realizada, con umbrales calculados sólo en TR.
  - En el ajuste fino los embeddings se pueden congelar.
- **Errores / edge cases:**
  - Sin ningún campo activo: error.
  - En el ajuste fino, unas cardinalidades distintas a las del modelo base (por ejemplo, otro número de tickers) dan error.

### 7. Capa oculta · `nn_hidden`

- **Propósito:** un bloque denso configurable, duplicable y encadenable.
- **Entradas:** `frame` con el estado de `nn_input` o de otra `nn_hidden`.
- **Configuración:**
  - `units`: **32 / 64 / 128 / 256** o custom (`custom_units`).
  - `name` y `repeat` (1–8).
  - `activation`, incluido `inherit`.
  - `dropout`, `norm` (none / batch / layer), `l1`, `l2` y `residual`.
  - Optimizador propio: `optimizer` (inherit / adam / adamw / sgd / rmsprop), `lr`, `weight_decay`, `momentum` y `clip`.
- **Salidas:** añade un bloque a `layers`. El informe muestra la tabla de todas las capas con formas, parámetros y optimizador.
- **Duplicación:**
  - **Clonar el nodo:** copiar y pegar en el lienzo. Cada instancia guarda sus propios parámetros, así que la configuración es
    independiente.
  - **Encadenar en serie:** `Oculta A → Oculta B → Oculta C`. Cada nodo añade su bloque a la pila en el orden del cable.
  - **Repeticiones:** `repeat = N` crea N bloques idénticos con pesos independientes.
  - **Herencia:** `activation = inherit` toma la activación de la capa anterior. `optimizer = inherit` usa el optimizador base de la
    capa de entrada.
- **Lógica:**
  - Orden del bloque: `Linear → Norm → activación → Dropout`, más el atajo residual (identidad, o una proyección lineal si cambian las
    unidades). L1/L2 se suman a la pérdida.
  - Un optimizador propio crea su propio grupo de parámetros. Si el tipo es distinto, se crea otro optimizador que avanza en el mismo
    paso.
- **Errores / edge cases:**
  - Unidades fuera de la lista: error.
  - Con BatchNorm se descartan los lotes de una sola fila.
  - En el ajuste fino, `freeze` deja la capa congelada.

### 8. Capa de salida · `nn_output`

- **Propósito:** la cabeza del modelo y la compilación.
- **Entradas:** `frame` con el estado de `nn_scaler` y la pila de capas.
- **Configuración:**
  - `task`: regression (h unidades, t+1..t+h), direction (h unidades, sigmoid) o multiclass (3 clases baja / lateral / sube con
    `flat_band`, softmax).
  - `activation`: linear, sigmoid, softmax o tanh.
  - `loss`: mse, mae, huber (`huber_delta`), bce o crossentropy.
  - `metrics` y `postprocess` (none / clip_quantile / sign).
  - Optimizador final (por defecto inherit) y `seed`.
- **Salidas:** `spec` con la arquitectura completa y la red instanciada (`Net`).
- **Lógica:**
  - Antes de entrenar hace un forward en seco con 4 filas para comprobar los shapes.
  - BCE y CrossEntropy reciben logits; sigmoid y softmax se aplican en `nn_predict`.
- **Errores / edge cases:**
  - Pérdida incompatible con la tarea (por ejemplo, MSE con direction): error.
  - Regresión con sigmoid o softmax: error.
  - Multiclase sin softmax o con h ≠ 1: error.
  - tanh sin MinMax [−1,1]: aviso.
  - Shapes incompatibles entre capas: error con el detalle de PyTorch.

### 9. Entrenar · `nn_train`

- **Propósito:** ajustar los pesos con selección en VA.
- **Entradas:** `frame` con el estado de `nn_output`.
- **Configuración:** `epochs`, `batch_size`, `patience`, `min_delta`, `scheduler` (none / plateau / cosine), `nan_policy`
  (reduce_lr / stop), `save_as`, `device` y `seed`.
- **Salidas:** la red en el mejor checkpoint y el historial de pérdidas. Con `save_as`, el archivo `artifacts/nn/models/<save_as>.pt`
  = `{spec, state, names, x_scaler, y_scaler}`.
- **Lógica:**
  - Por lote: `zero_grad → forward → pérdida + L1/L2 → backward → clip_grad_norm_ → step` de cada optimizador.
  - Checkpoint en la época con menor pérdida en VA, y early stopping.
- **Errores / edge cases:**
  - Pérdida o gradiente NaN/Inf: restaura el mejor checkpoint y divide el lr a la mitad, como máximo 3 veces; luego para.
  - Divergencia desde la primera época: error que sugiere bajar el lr, activar clipping o revisar el escalado.

### 10. Ajuste fino · `nn_finetune`

- **Propósito:** reutilizar un modelo preentrenado en otro activo o en un periodo reciente.
- **Entradas:** `frame` de la rama nueva (con el estado de `nn_scaler`) y el pin `pretrained` (la salida de `nn_train` de la rama
  base). Alternativamente, un archivo.
- **Configuración:**
  - `source` (pin / file) y `load`.
  - `freeze`: none, `hidden_bottom` (entrada + embeddings + las N primeras ocultas, con `freeze_layers`), `all_but_output` o
    `embeddings`.
  - `lr_factor`, que multiplica todos los learning rates.
  - `epochs`, `patience`, `save_as` y `device`.
- **Salidas:** la red ajustada, compatible con `nn_predict`. El informe compara la pérdida en VA del modelo base sin ajustar con la del
  modelo ajustado, y muestra los parámetros entrenables frente a los totales.
- **Lógica:**
  - Copia la arquitectura y los pesos, pone `requires_grad = False` en las capas congeladas, reduce el lr y entrena con early stopping.
  - Usa el escalador de la propia rama, o el guardado si se carga con `load_transform`.
- **Errores / edge cases:**
  - Pin sin conectar: error.
  - Número de rasgos distinto al del modelo base: error.
  - Modelo base con embeddings y rama sin embeddings, o con cardinalidades distintas: error.
  - Divergencia: error que sugiere bajar `lr_factor`.

### 11. Predicción e inferencia · `nn_predict`

- **Propósito:** predecir en TR/VA/TE y en vivo, en unidades reales.
- **Entradas:** `frame` con el estado de `nn_train` o `nn_finetune`.
- **Configuración:** `interval` (nivel de la banda) y `export` (nombre del CSV).
- **Salidas:**
  - El `frame` con las columnas `Pred`, `Lower` y `Upper` (si hay un solo ticker).
  - Una tabla de métricas por bloque:
    - Regresión: RMSE, MAE, R², acierto direccional, RMSE del pronóstico ingenuo 0 y cobertura de la banda.
    - Clasificación: accuracy frente a la clase mayoritaria.
  - Las filas en vivo: la predicción para t+1..t+h y, si el objetivo es LogReturn, el precio estimado.
  - El archivo `artifacts/nn/exports/<export>.csv`.
- **Lógica:**
  - `inverse_transform` del escalador del objetivo.
  - Banda conformal split: cuantil α de |residuo| en VA.
  - TE se mide una sola vez.
- **Errores / edge cases:**
  - Sin entrenamiento aguas arriba: el lienzo lo marca como error.
  - Cuando no hay filas en vivo, la tabla lo indica vacía.

---

## 3 · Ejemplo completo multicapa

**Nombre: `nn-multicapa-yfinance`**
(«Redes Neuronales · MLP multicapa 128 → 64 → 32 (yfinance, embeddings, ajuste fino)»). El archivo es
`public/blueprints/nn-multicapa-yfinance.json` y la entrada está en la biblioteca. Lo genera `python scripts/build_nn_blueprints.py`.

- **Fila 1 (SPY, 10 años):**
  - Datos → Guardia → Entrada (LogReturn, Range y LogVolume × 10 lags = 30 rasgos, AdamW lr 1e-3) → TR/VA/TE 70/15/15 → Escalador
    Standard persistido → Embeddings (día, mes y régimen, dim 4, concat).
  - **Oculta A** 128 GELU LayerNorm Dropout 0.1 → **Oculta B** 64 inherit residual L2 → **Oculta C** 32 SiLU con Adam lr 5e-4 propio.
  - Salida: regresión t+1, linear, Huber.
  - Entrenar: 60 épocas, plateau y early stopping; se guarda `spy_mlp_base`.
  - Predicción con banda del 90 % y exportación a CSV.
- **Fila 2 (QQQ, 5 años):** la misma entrada y los mismos embeddings. **Ajuste fino** desde el pin `pretrained` (congela la entrada y la
  Oculta A, lr × 0.1) y predicción en vivo.

Resultado verificado con datos reales (25-nov-2026): SPY en TE tiene RMSE 0.00883 frente a 0.00868 del pronóstico ingenuo, acierto
direccional del 50.3 % y cobertura de la banda del 93.6 %. La red **no** supera al pronóstico ingenuo: el retorno diario es casi
impredecible. El ejemplo demuestra el flujo; no es una estrategia.
