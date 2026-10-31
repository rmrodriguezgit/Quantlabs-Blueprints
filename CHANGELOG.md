# Changelog

## v0.3.1 — 2026-10-31

- Columna CumulativeLogReturn calculada como la suma acumulada de los LogReturn retenidos tras el calentamiento.
- Disponible en Plot y en la selección de características de entrada.


## v0.3.0 — 2026-10-31

- Monitor de ejecución con nodo activo, tiempo transcurrido, filas disponibles, progreso por nodos y pérdidas por época. Electron y navegador reciben eventos WebSocket.
- Corrección de diagnóstico para EURUSD=X y otros activos sin volumen: VWAP produce un mensaje explícito para desactivarlo en lugar de eliminar silenciosamente todas las filas.
- Errores de preparación con cantidades de filas, secuencias, divisiones y recomendaciones concretas; se distingue ejecución detenida de cálculo activo.
- Resumen visible de arquitectura y selección de características. Entrada sincronizada automáticamente y comprobación previa de incompatibilidades.
- Nodo Plot configurable: datos/indicadores, TR/VA/TE, pronósticos, pérdidas, equity y señales, con líneas, áreas o barras.
- Ejemplo completo de red 16 → 64 ReLU → 32 ReLU → 1 con gráficos de indicadores y pérdidas.
- README con banner, badges y enlace a versión etiquetada en GitHub.

## v0.2.0 — 2026-10-31

- Indicadores seleccionables y configurables: VWAP, SMA, EMA, MACD, RSI y bandas de Bollinger.
- Selección de características de entrada; arquitectura compuesta por Network Input, capas y Network Output.
- Duplicación, ramificaciones, unión de características y ejemplo 16 → 64 → 32 → 1.

## v0.1.0 — 2026-10-31

- Base Electron/React/Python con lienzo Blueprint, modelos PyTorch, API, backtest, importación y exportación JSON.
