# Changelog

## v0.5.6 — 2026-11-02

- Scatter con X vacío detecta Coordinate en Monthly Curve Data y coordinate en informes Curve Fitting, conservando X explícita y fechas en líneas.
- Monitor de épocas muestra «sin validación separada» cuando no hay pérdida VA, en lugar de undefined.


## v0.5.5 — 2026-11-02

- Doble clic o Enter en conectores: alternativas compatibles existentes y creación de nodos conectados desde salidas.
- Desconexión por enlace, conector o nodo completo sin eliminar nodos; clic en cable muestra Desconectar.
- Validación compartida con arrastre: tipos, ciclos, entradas ocupadas y entradas exclusivas; acciones desactivadas durante ejecución.


## v0.5.4 — 2026-11-01

- Ver JSON, Editar, Guardar y Borrar en cada ejemplo incluido y personal. Editor con validación de grafo y protección de cambios pendientes.
- Cambios locales a plantillas incluidas sin duplicarlas; actualización de diseños personales y opción de usar el lienzo actual.
- Papelera recuperable para ejemplos incluidos y personales; restauración conserva nombre, parámetros y conexiones.
- Persistencia con recuperación del estado anterior si falla una escritura; pruebas de JSON, almacenamiento y acciones de biblioteca.

## v0.5.3 — 2026-10-31

- New en barra y File → New (⌘N / Ctrl+N): lienzo vacío persistente, archivo automático del diseño actual en Personal y ejecución bloqueada mientras esté vacío.

- Herramientas faltantes del notebook: Swish en Network Layer y conexión de arquitecturas moradas al entrenador Curve Fitting; sincronización de entradas 1 / grado polinómico.
- Resumen de capas ejecutadas, formas y parámetros en el informe de curvas, conservando estados BatchNorm y aleatoriedad.
- Plot con puntos, puntos + curva suave de 300 coordenadas, eje X numérico seleccionable, Y logarítmica y descarte inicial del historial.
- Las tres fichas existentes exponen la arquitectura de alta capacidad; se mantienen 18 fichas y se reutilizan los cálculos de datos, escalado, regularización y entrenamiento.


## v0.5.2 — 2026-10-31

- Auditoría de escalado por nodo: método, ámbito/filas de ajuste, centro/divisor y estadísticas antes/después; distingue X, y, columnas constantes y ramas sin escalado.
- Corrección del checkbox normalize en polinomios: ahora puede desactivar realmente StandardScaler. Todas las redes de la ficha temporal EUR/USD se normalizan; las fichas originales conservan sus opciones.
- Ajuste explícito de PolynomialFeatures con TR, validación de matrices finitas/nombres de escalador y métricas de curvas tras invertir el centrado.
- Pruebas de límites temporales, columnas constantes, ventanas sin duplicación, folds HPO, objetivo en unidades originales e invariancia al cambiar −2…2 por −1…1.


## v0.5.1 — 2026-10-31

- Blueprint del notebook documentado EUR/USD: fuente de 15 años, cierre mensual y coordenada artificial; siete ramas OLS/neuronales con benchmark conjunto.
- Tres fichas: inicio rápido, épocas/arquitecturas originales y evaluación temporal separada. Biblioteca ampliada a 18 diseños.
- Nodos Monthly Curve Data y Curve Fitting / Benchmark: activaciones, optimizadores, L2, ReduceLR, épocas y protocolos configurables.
- Métricas vectoriales sin broadcasting, RMSE en pips y tiempos; guías, mapa de celdas y notebook acompañante ejecutable con el motor del proyecto.


## v0.5.0 — 2026-10-31

- Biblioteca integrada con diez infografías originales y 15 blueprints ejecutables, organizados por área.
- Cargar/editar/ejecutar, guardar copias personales, archivar el diseño anterior e importar/exportar un diseño o la colección completa; formato JSON versionado compatible con grafos anteriores.
- Nodo Learning Experiment: comparación de SGD/RMSprop/AdamW, regularización/dirección, HPO temporal, CNN de dos capas, LSTM/GRU/bidireccionales, PCA/AE/DAE, CVAE/cGAN/cDCGAN multiactivo y Double DQN.
- Protocolos intradía en bps separados del objetivo close-to-close anterior; selección por validación, costes explícitos y aislamiento de test.
- Informes por experimento con tablas, métricas y gráficos seleccionables desde Plot; dimensiones de entrada visibles y derivadas de características.
- Corrección de los estados finales bidireccionales en el entrenador general.
- Guía de uso, alcance y procedencia; hashes SHA256 de imágenes, pruebas de todas las plantillas, persistencia/importación y ausencia de selección con test.


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
