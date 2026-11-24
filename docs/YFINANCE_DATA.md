# yfinance Data

Nodo fuente (**01 / DATA**) que descarga un ticker de Yahoo Finance exponiendo las opciones de `yfinance.download` (verificado con yfinance 0.2.66). Código: [`backend/engine/nodes/yfinance_nodes.py`](../quant-blueprint-studio/backend/engine/nodes/yfinance_nodes.py).

| Parámetro | Valores | Notas |
| --- | --- | --- |
| `ticker` | un símbolo de Yahoo | `SPY`, `AAPL`, `BTC-USD`, `EURUSD=X`, `^GSPC`… |
| `interval` | 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo | |
| `range_mode` | `period` / `start_end` | el Inspector muestra sólo los campos del modo elegido |
| `period` | 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max | |
| `start`, `end` | AAAA-MM-DD | `end` es **exclusivo** (como en yfinance); vacío = hoy |
| `auto_adjust` | sí/no | ajusta OHLC por dividendos y splits; sin él aparece `AdjClose` |
| `back_adjust` | sí/no | sólo con `auto_adjust` desactivado |
| `actions` | sí/no | añade `Dividends` y `StockSplits` (y `CapitalGains` en fondos) |
| `prepost` | sí/no | pre y post mercado, sólo intradía |
| `repair`, `keepna`, `rounding` | sí/no | opciones de yfinance |
| `drop_incomplete` | sí (por defecto) | descarta la última barra si aún no ha cerrado (minuto en curso o vela diaria de hoy) |
| `timezone` | America/New_York, UTC, … | el intradía llega en UTC y se expresa en esta zona |
| `timeout`, `cache_seconds` | | caché de intradía máx. 60 s |

**Límites de Yahoo (medidos en vivo):** Yahoo no da error fuera de estos rangos, devuelve 0 filas. El nodo y la validación del lienzo lo comprueban antes de descargar y proponen un periodo válido.

| Intervalo | Máximo |
| --- | --- |
| 1m | ~7 días por descarga, sólo los últimos 30 días |
| 2m, 5m, 15m, 30m, 90m | 60 días |
| 60m, 1h | 730 días |
| 1d y mayores | sin límite |

El informe incluye barras, rango, ajuste aplicado, retorno total, volatilidad anualizada (según el intervalo), caída máxima, máximos/mínimos, barras sin volumen (avisa en divisas/índices y en pre/post mercado), acciones corporativas, gráficos de cierre y volumen y las últimas barras. La salida es un `frame` OHLCV compatible con Technical Features, Data Quality, Laya Signal (su nombre de activo `auto` toma este ticker), Plot, etc. Ejemplo: **«yfinance Data · intradía, ajustes y acciones corporativas»**.
