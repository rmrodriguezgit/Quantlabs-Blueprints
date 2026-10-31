# Changelog

## v0.4.1 — 2026-10-31

- Panel de indicadores del portafolio completo con 11 tarjetas y selector de cartera; pesos y referencia visibles.
- Métricas de cartera, activos y benchmark separadas en la API, conservando la tabla anterior y compatibilidad con resultados previos.
- Explicaciones para Sharpe, Sortino y OLS no definidos.
- Validación del retorno agregado frente a promedios incorrectos y prueba de renderizado de todas las tarjetas.


## v0.4.0 — 2026-10-31

- Plot con títulos, origen, identificación de nodo, paleta de ocho colores, colores por serie y leyendas personalizadas. Cambios de presentación inmediatos sin reentrenar.
- Nodos Portfolio Data, Portfolio / Markowitz y Risk Metrics, con pines de portafolio independientes.
- Hasta 20 activos, benchmark, pesos personalizados, Markowitz de máximo Sharpe y mínima varianza, pesos iguales y evaluación temporal separada.
- ROI, CAGR, VaR/CVaR histórico, Sharpe, Sortino, alpha/beta OLS, volatilidad y drawdown, con convenciones visibles.
- Estimación de covarianza con shrinkage o muestra, límite de peso, rebalanceo diario o buy-and-hold, costes de entrada/rebalanceo/salida.
- Panel de métricas/pesos y gráficos de equity, log-retorno acumulado, drawdown, precios y pesos.


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
