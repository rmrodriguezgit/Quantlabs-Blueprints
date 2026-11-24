# Polymarket × LAYA · decisión UP / DOWN

Dos nodos nuevos en la categoría **10 / POLYMARKET**:

| Nodo | Kind | Entrada | Salida |
| --- | --- | --- | --- |
| Polymarket Market | `polymarket_source` | — | `frame` |
| LAYA UP/DOWN | `laya_updown` | `frame` | `frame` + informe con la decisión |

Código: [`backend/engine/nodes/polymarket_nodes.py`](../quant-blueprint-studio/backend/engine/nodes/polymarket_nodes.py), Inspector: [`PolymarketPicker.tsx`](../quant-blueprint-studio/src/renderer/components/PolymarketPicker.tsx).

## Endpoints verificados (lectura pública, sin clave)

| Uso | Endpoint |
| --- | --- |
| Metadatos por slug / condition_id / token | `GET gamma-api.polymarket.com/markets?slug=` · `?condition_ids=` · `?clob_token_ids=` |
| Mercados de un evento | `GET gamma-api.polymarket.com/events?slug=` |
| Búsqueda de texto | `GET gamma-api.polymarket.com/public-search?q=&limit_per_type=` |
| Historial de P(token) | `GET clob.polymarket.com/prices-history?market=<token>&interval=&fidelity=` → `{history:[{t,p}]}` |
| Libro de órdenes | `GET clob.polymarket.com/book?token_id=` → `bids/asks [{price,size}]` |

Gamma devuelve `outcomes`, `outcomePrices` y `clobTokenIds` como cadenas JSON. Los mercados recurrentes de cripto usan el slug `<activo>-updown-<5m|15m|1h>-<inicio unix>` y etiquetas `5M`/`15M`/`1H`. A fecha de verificación existen 5m, 15m y 1h para BTC; **30m no se publica**, así que esa ventana sólo se usa como sobrescritura manual.

Llamadas con caché (`cache_seconds`), timeout por petición, 2 reintentos con espera en 5xx/429/red, y si falla se reutiliza la última respuesta en caché. Los errores se reportan en el nodo, sin colgar el grafo.

## Precio del subyacente y precio a vencer

Los mercados `<activo>-updown-*` se resuelven con el **TWAP de Chainlink** (p. ej. BTC/USD): UP si el precio al final de la ventana es ≥ al del inicio. Gamma no trae ese precio, así que el nodo lo obtiene de:

| Dato | Fuente verificada |
| --- | --- |
| Precio a vencer (apertura) | `GET polymarket.com/api/crypto/crypto-price?symbol=BTC&eventStartTime=&variant=&endDate=` → `openPrice`. Variantes válidas: `fiveminute` (5m), `fifteen` (15m), `onehour` (1h). Otras variantes devuelven un valor genérico y no se usan; **30m no tiene variante** |
| Precio actual | WebSocket `wss://ws-live-data.polymarket.com`, tema `crypto_prices_chainlink`, filtro `{"symbol":"btc/usd"}`: el primer mensaje trae los últimos segundos (1 punto/s) |

El dashboard muestra un panel con precio a vencer, precio actual, diferencia ($ y %), quién ganaría si cerrara ahora, cuenta atrás y un gráfico del precio con la línea del objetivo; el registro de ciclos añade el subyacente y Δ objetivo. Si alguna fuente falla, el panel muestra «—» y el grafo sigue. Mercados no cripto no muestran el panel.

### Señal `z_target` (distancia al precio a vencer)

```
z_target = clip( Δ% desde la apertura de la ventana / (σ 1m · √minutos restantes), −6, 6 )
```

σ = desviación de los retornos de 1 minuto de los 30 minutos previos a la apertura (mínimo 0.005 %/min). En paseo aleatorio P(UP) ≈ Φ(z) ≈ σ(1.7·z). Se calcula con **velas de 1 min de Binance** (`api.binance.com/api/v3/klines`, BTC/ETH/SOL/XRP/DOGE/BNB) tanto en vivo como para cada ventana histórica, para que calibración y decisión vean la misma magnitud: el nivel de Binance difiere unos dólares del de Chainlink, pero el movimiento dentro de la ventana coincide (verificado: +$67 Binance vs +$71 Chainlink en la misma ventana). En histórico sólo se usan velas cerradas antes del momento de decisión; antes de que abra la ventana vale 0. También se envía a Laya (`change_since_window_open_pct`, `distance_to_price_to_beat_z`).

Peso en el Inspector: `w_target` (1.0). Laya Calibrate lo ajusta y compara la señal sola («distancia objetivo») con mercado, Laya cruda y calibrada. Los estados online anteriores incorporan el peso nuevo con el valor del Inspector.

Prueba en vivo (120 ventanas BTC 5m, decisión al minuto 1; TE = 24 ventanas, muestra pequeña):

| TE | mercado | Laya cruda | distancia objetivo | calibrada |
| --- | --- | --- | --- | --- |
| accuracy | 0.625 | 0.542 | **0.750** | 0.708 |
| log-loss | 0.623 | 0.794 | 0.539 | **0.522** |

Pesos calibrados: target 1.17, market 0.27, laya 0.21, momentum 0.09, bias 0.15. Con 24 ventanas de prueba la diferencia de acierto aún puede ser azar; el log-loss es la comparación más fiable.

## Polymarket Market · Inspector

- **Buscador** (Gamma `public-search`) → clic en un resultado lo selecciona y lo valida.
- **↻ Validar / refrescar**: resuelve el mercado y muestra outcomes UP/DOWN, P(UP), ventana detectada, estado y liquidez.
- `market`: slug, URL (`polymarket.com/event/...`), `condition_id` (0x…64), `token_id` o texto. Un alias sin timestamp (`btc-updown-15m`) apunta a la ventana activa ahora.
- `window`: `auto | 5m | 15m | 30m | 1h` (auto = slug → etiquetas → duración inicio-cierre); `fallback_window` si no se detecta.
- `up_outcome`: `auto` mapea Up/Yes/Higher/Above → UP y Down/No/Lower/Below → DOWN; `0`/`1` fuerza el índice.
- `history_interval` (`1h|6h|1d|1w|max`), `fidelity` (min), `cache_seconds`, `timeout`.

Columnas de salida: `P_up`, `P_down`, `Close` (= P_up, para reutilizar Laya Signal, Plot…), `Open/High/Low/Volume`, `SecondsToClose` y `Book_best_bid/best_ask/spread/bid_depth/ask_depth/imbalance`. El libro es un snapshot actual y **sólo se escribe en la última fila** (sin mirada al futuro). Metadatos en `frame.attrs['polymarket']`.

## LAYA UP/DOWN

```
P(UP) = σ( bias + w_market·logit(P_up) + w_momentum·z_mom + w_laya·log(P_up_laya/P_down_laya) + w_imbalance·2·imbalance )
```

- `z_market`: mid del libro si el spread ≤ 0.10, si no el último punto del historial.
- `z_mom = 2·tanh(Δp / (σ·√n))` sobre `lookback_bars`.
- `z_laya`: si hay **Laya Signal** aguas arriba usa su última fila; si no, y `use_laya=true`, envía a `laya-mlx` (reutiliza `laya_nodes._agent`) un estado con P implícita, momentum, libro, spread, minutos al cierre y horizonte, con una pregunta `choice` up/down.
- **Pesos**: `weight_source=manual` usa los sliders `w_*`/`bias` del Inspector; `calibrated` toma `bias` y el peso del canal Laya de **Laya Calibrate** (`attrs.laya_weights`). Validación avisa si falta Calibrate.
- `window=inherit` usa la ventana del mercado; puede fijarse `5m|15m|30m|1h`.
- `min_confidence` marca la decisión como `actionable`.

### Formato de salida

`result.decisions[]` (y nota JSON en el informe del nodo):

```json
{
  "id": "UD", "decision": "UP", "probability_up": 0.7564, "confidence": 0.7564, "actionable": true,
  "horizon": "1h", "market": "btc-updown-1h-1795374000", "as_of": "2026-11-22 19:11:13",
  "weights": {"market": 1, "momentum": 0.5, "laya": 1, "imbalance": 0.3, "bias": 0},
  "weight_source": "Inspector (manual)",
  "signals": {"market": 0.02, "momentum": 0.0, "laya": 2.17, "imbalance": -0.55},
  "laya": {"used": true, "via": "laya-mlx directo", "P_up": 0.898, "P_down": 0.102}
}
```

El frame añade `UpDown_P_up` y `UpDown_Decision` (+1/−1) en la última fila. La tabla del informe es la traza por señal (z, peso, contribución); el gráfico muestra la contribución en log-odds.

## Aprendizaje en línea de pesos (`weight_source = online`)

1. **Serie**: todas las ventanas de una serie comparten pesos (`btc-updown-5m-1795375800` → `btc-updown-5m`; o `learning_key`).
2. **Registro**: en cada ventana abierta se guarda la decisión (z, P(UP), P mercado). `commit_at=first` guarda la primera decisión de la ventana (la más honesta: cerca del cierre el precio ya delata el resultado); `last` la sobrescribe en cada ciclo.
3. **Resolución**: en cada ciclo se consultan las ventanas pendientes en Gamma (`/markets?slug=&closed=true`, porque Gamma oculta los cerrados por defecto). Cuando `closed=true` y un outcome vale 1 → y = 1 (UP) o 0 (DOWN). Los 5m se resuelven en minutos; las ventanas sin resolver en 6 h se descartan; los 50-50 no se aprenden.
4. **Actualización** (SGD sobre log-loss): `w ← clip(w − lr·((p − y)·z + L2·w), −5, 5)` y `bias ← bias − lr·(p − y)`.
5. **Persistencia**: `.quantlabs/polymarket/<serie>.json` (ignorado por git; `QL_LEARNING_DIR` lo cambia). Sobrevive a reinicios. **Reiniciar** en el Inspector lo borra y el siguiente ciclo parte de los sliders.

El dashboard muestra ventanas resueltas y pendientes, acierto de LAYA frente al del mercado (P ≥ 0.5) en las últimas 50, log-loss, pesos inicio → ahora, su evolución y la tabla de ventanas aprendidas. El ejemplo 5m viene configurado en `online` con `w_laya = 0.3` y `lookback_bars = 5`.

Limitaciones: con un `lr` alto los pesos persiguen el ruido; con pocas decenas de ventanas el acierto no es estadísticamente significativo. Compáralo siempre con el acierto del mercado.

## Laya Signal y Laya Calibrate con ventanas resueltas

Tras **Polymarket Market**, los nodos Laya de la categoría 09 cambian solos de modo (`mode = auto`):

- **Laya Signal (modo Polymarket)**: toma las últimas `train_windows` ventanas ya resueltas de la serie (`btc-updown-5m-*`), reconstruye el estado de cada una a `decision_offset_s` segundos del inicio con `prices-history?startTs&endTs` (lookback `pm_lookback` puntos de 1 min) y pregunta a Laya «¿UP o DOWN al final de la ventana?». Usa **las mismas funciones** (`market_signals`, `laya_state`, `laya_question`) que LAYA UP/DOWN en vivo. Las ventanas resueltas quedan en `.quantlabs/polymarket/windows/` (inmutables). La última fila es la ventana actual, con libro.
- **Laya Calibrate (modo Polymarket)**: regresión logística `P(UP) = σ(bias + w_market·z_market + w_momentum·z_momentum + w_laya·z_laya)` sobre el resultado real; orden cronológico TR/VA/TE, época 0 = seguir al mercado. Informa accuracy, Brier, log-loss y ECE en TE para *mercado*, *Laya cruda* y *calibrada*, más el diagrama de fiabilidad. El desequilibrio no tiene histórico: su peso se fija con `imbalance_weight` (0) y lo aprende el modo online.
- **LAYA UP/DOWN**: `calibrated` usa esos pesos; `online` parte de ellos en una serie nueva (`seeded_from: Laya Calibrate`) y los sigue corrigiendo con cada ventana resuelta.
- **Laya Feedback** sigue siendo para estrategias diarias; con Polymarket la validación y el motor lo rechazan con un mensaje que indica usar el modo online.

Prueba en vivo (2026-11-22, 120 ventanas btc 5m): tasa base UP 0.558, acierto de Laya cruda 0.558 (casi siempre dice UP), mercado al minuto 1 0.583; en TE la calibrada iguala al mercado (0.583) con log-loss 0.636 vs 0.630. Es decir: con estos datos Laya no añade información sobre el precio; la calibración lo detecta y le asigna un peso pequeño (~0.1).

## Predicción por minuto dentro de la ventana

Los pesos óptimos cambian con el tiempo restante (al inicio pesan mercado y momentum; al final, la distancia al precio a vencer). Por eso hay **un juego de pesos por tramo**:

- **Laya Signal** `bucket_s`: `auto` = 60 s para 5m/15m y 300 s para 30m/1h; `0` = un único momento (`decision_offset_s`). Cada ventana resuelta se reconstruye en el **punto medio de cada tramo** (5m: 0:30, 1:30, 2:30, 3:30, 4:30) sólo con datos anteriores a ese instante. Cada ventana se descarga una vez (historial CLOB y velas completos) y se guarda en disco; los tramos se recortan de ahí.
- **Laya Calibrate** ajusta una regresión por tramo. **TR/VA/TE se reparten por ventana** (orden cronológico), así los cinco tramos de una misma ventana nunca caen en conjuntos distintos. La tabla da, por minuto: TE n, acierto de mercado/distancia/calibrada, log-loss de mercado/distancia/calibrada y **ventaja vs mercado** = log-loss mercado − log-loss calibrada (positivo = el modelo es más preciso que el precio en ese minuto). Gráficos: log-loss por minuto y pesos por minuto.
- **LAYA UP/DOWN** calcula el minuto actual (`tiempo transcurrido // bucket_s`) y usa sus pesos (`calibrated`) o su juego online (`online`, clave `slug#minuto`: una decisión registrada por minuto y ventana; cada una corrige sólo los pesos de su minuto). Si un minuto no tuvo datos para calibrar, usa el más cercano. `minute_buckets = false` vuelve a un único juego.
- **Dashboard** «Por minuto de la ventana»: curva de P(UP) de LAYA y del mercado dentro de la ventana actual (con divisiones por minuto), barras de ventaja por minuto (minuto actual en ámbar) y tabla con calibración TE y acierto online por minuto.

El ejemplo `polymarket-laya-calibrated-5m` («… 5m (por minuto)») ya viene configurado así, con la serie online `btc-updown-5m-por-minuto`.

## Explicación automática de los reportes

`brain/explain.py` tiene lecturas específicas para Polymarket (antes caían en la genérica o leían claves del modo diario): Polymarket Market (P(UP), spread, desequilibrio, subyacente vs precio a vencer y z), Laya Signal por minuto (acierto mercado/Laya/distancia y aviso si Laya ≈ tasa base), Laya Calibrate (compara contra **el mercado**, minuto con mayor ventaja, en cuántos minutos gana, aviso de muestra pequeña) y LAYA UP/DOWN (señal dominante, señales en contra, acuerdo/desacuerdo con el mercado, ventaja medida en el minuto actual, acierto online).

## Polymarket Dashboard (`polymarket_dashboard`)

Nodo fuente sin entradas que descarga **los eventos activos** de la Gamma API (`/events?active=true&closed=false&order=…&limit=50&offset=…`; Gamma sirve páginas de más de 50 sólo comprimidas, por eso pagina de 50 en 50) y los organiza en un dashboard:

| Bloque | Contenido |
| --- | --- |
| KPIs | volumen 24 h, interés abierto, liquidez, eventos, mercados abiertos, cierran en 24 h, disputados (40–60 %), spread mediano, concentración de los 10 eventos más activos |
| Categorías | volumen 24 h, eventos e interés abierto por categoría (sin etiquetas genéricas como *Recurring*); con una categoría elegida, por subcategoría (Bitcoin, Ethereum…) |
| Rankings (pestañas) | más negociados, mayores movimientos 24 h, más disputados, cierran en 24 h, más líquidos, spreads amplios. Movimientos y disputados excluyen mercados casi resueltos (< 3 % o > 97 %) |
| Otros | eventos más activos, distribución de probabilidades, resumen de mercados Up/Down recurrentes |

### Conectado a Polymarket Market: dashboard de un mercado

El nodo tiene una entrada `frame` **opcional**. Si se conecta a **Polymarket Market** (directamente o a través de un nodo que conserve sus datos, como LAYA UP/DOWN), muestra el dashboard de *ese* mercado; sin conexión, la vista general de abajo.

| Bloque | Fuente |
| --- | --- |
| Outcomes con probabilidad (libro), VWAP reciente, cuenta atrás, subyacente vs precio a vencer | Polymarket Market + CLOB `/book` |
| KPIs: spread, cambio 1 h / 24 h, rango, volumen reciente y de Gamma, liquidez, presión hacia cada outcome, operaciones/min, operación media y mayor, profundidad ±5¢, desequilibrio | CLOB + Data API v2 |
| Historial de P, profundidad acumulada del libro, flujo de operaciones por minuto, libro de 12 niveles de cada outcome | CLOB `/prices-history`, `/book` |
| Últimas operaciones (sin wallets ni nombres) | Data API v2 `GET /v2/trades?condition=<condition_id>` |
| Otros mercados del mismo evento (elecciones, rangos de precio…) | Gamma `/events?slug=` |
| Concentración de tenedores: mayor y top 5 respecto a los 20 mayores (sólo agregados) | Data API v2 `GET /v2/holders?condition=` |
| Decisión LAYA UP/DOWN, si está en la cadena | atributos del frame |

**Presión hacia un outcome** = comprar ese outcome o vender el contrario: el lado del *taker* por sí solo engaña en un mercado binario (comprar «Down» es apostar contra «Up»). El volumen de Gamma y el historial CLOB pueden ir con retraso; los KPIs usan el libro y las operaciones. Al final de una ventana Up/Down Polymarket retira las órdenes y el libro queda vacío. Con este dashboard conectado y sin LAYA UP/DOWN, el monitor UP/DOWN no se muestra (sería redundante). Ejemplo: **«Polymarket · dashboard de un mercado (BTC 5m)»**.

### Sin conexión: todo Polymarket

Inspector: categoría, orden (`volumen_24h`, `liquidez`, `nuevos`), texto en el título, páginas (1–10), filas por ranking, volumen mínimo, refresco en vivo (60 s), caché y timeout. Cada mercado enlaza a su página de Polymarket (se abre en el navegador del sistema; la app sólo permite `https://polymarket.com/event/…`). La salida `frame` es la tabla de mercados (para Plot u otros nodos) y el informe incluye una explicación automática. Ejemplo: **«Polymarket Dashboard · todo el mercado activo»**.

Prueba en vivo (2 páginas, todas las categorías): 100 eventos, 5 235 mercados, $31.7 M de volumen 24 h en 4.5 s; Sports lidera el volumen.

## Ejemplos (Biblioteca → Polymarket · decisiones UP/DOWN)

| Ejemplo | Mercado | Ventana |
| --- | --- | --- |
| `polymarket-updown-5m` | `btc-updown-5m` | auto → 5m |
| `polymarket-updown-15m` | `btc-updown-15m` | auto → 15m |
| `polymarket-updown-30m` | `eth-updown-1h` | manual 30m |
| `polymarket-updown-1h` | `btc-updown-1h` | auto → 1h |
| `polymarket-dashboard` | eventos activos de Polymarket | — |
| `polymarket-market-dashboard` | Polymarket Market `btc-updown-5m` → Polymarket Dashboard | auto → 5m |
| `polymarket-laya-calibrated-5m` | `btc-updown-5m` → Laya Signal (por minuto) → Laya Calibrate (pesos por minuto) → UP/DOWN online por minuto | auto → 5m |

Verificación en vivo (2026-11-22): los cuatro ejecutan y devuelven decisión; tests offline con HTTP simulado en `backend/tests/test_polymarket.py`.

## Advertencias

- Laya no se entrenó con mercados de predicción; en las pruebas su P_up sale alta (~0.9) casi siempre, lo que domina con `w_laya=1`. Bájalo o calíbralo antes de confiar en él.
- Es una señal de investigación, no asesoría financiera. Los nodos sólo leen; no envían órdenes.
