# Verificación del proyecto (v0.5.2)

Fecha: 31 de octubre de 2026. Entorno local macOS, Python 3.12.12, Node.js 22.22.3, npm 10.9.8.

## Auditoría de StandardScaler (v0.5.2)

- 50 pruebas Python aprobadas (14 nuevas de escalado), compilación TypeScript/Vite/Electron y suites de grafos/biblioteca/portafolio UI/escalado UI correctas.
- Ajustes comprobados sólo con TR, límites exactos de barras antes de expandir ventanas, columnas constantes, transformaciones de VA/TE sin reajuste, y escaladores nuevos por fold HPO.
- Invariancia de predicciones y MSE al cambiar −2…2 por −1…1 con StandardScaler, tanto en red densa como polinomio neuronal.
- Corrección del checkbox polinómico comprobada capturando llamadas reales a StandardScaler.fit: desactivado no ajusta scaler; activado recibe únicamente columnas polinómicas TR.
- Diagnósticos distinguibles para StandardScaler, centrado de y, MinMax y ramas desactivadas. Compatibilidad con resultados antiguos sin auditorías.
- Interfaz ejecutada con la ficha temporal EUR/USD real: 12 nodos completados, seis ramas neuronales escaladas. Auditoría muestra 117 filas TR hasta 2021-06-30, media transformada 2.29636e−16 y desviación 1 para Coordinate.
- Documentación en docs/SCALING.md; captura docs/v0.5.2-scaler-audit.png. Los grafos personales existentes conservan su selección de normalización; cargar la ficha temporal actualizada activa las seis ramas neuronales.

## Blueprint del notebook EUR/USD (v0.5.1)

- 36 pruebas Python y suites TypeScript/grafos/portafolio UI/biblioteca aprobadas; 18 diseños válidos y compilación Electron/Vite correcta.
- Remuestreo al último cierre mensual, coordenada artificial, dimensiones 769/6 parámetros y protección contra broadcasting comprobados; cambios sólo TE no modifican media TR ni checkpoint neuronal temporal.
- Descarga real EURUSD=X de 15 años: 181 meses, 2011-10-31 a 2026-10-31. Las siete ramas del modo rápido (100 épocas) completadas en CLI y Electron; benchmark y gráficos visibles.
- Ficha de referencia con datos reales: completó 1500/1500/1500/1500/2000/2500 épocas en las seis redes; MSE descriptivo de la DNN de tres capas: 0.0002478107. Este error es in-sample y no se interpreta como pronóstico.
- Notebook acompañante: todas sus celdas de código ejecutadas con datos sintéticos, tabla de siete modelos, SVG y exportaciones JSON/CSV comprobadas. `ipykernel` opcional instalado en la .venv; `pip check` correcto.
- Original adjunto leído sin ejecución ni modificaciones. Procedencia y mapa de celdas en docs/NOTEBOOK_CURVE_SOURCE.json; alcance en docs/NOTEBOOK_EURUSD.md.
- No se afirma equivalencia numérica entre PyTorch y TensorFlow/Keras. Los modos in_sample son ajustes descriptivos; la extensión temporal evalúa extrapolación.

## Biblioteca y experimentos (v0.5.0)

- TypeScript estricto, build Vite/Electron y suites de grafos, portafolio UI y biblioteca: correctos.
- 32 pruebas Python, incluyendo ejecución de las 15 plantillas con presupuesto abreviado (2 épocas / 2 trials), invariancia de selección frente a cambios sólo en test, alineación de objetivo intradía y dimensiones de cDCGAN/Double DQN.
- Ejecución adicional de las 15 plantillas con sus parámetros predeterminados (12 épocas máximas, 6 trials HPO), completada con datos sintéticos y resultados JSON finitos.
- Las 10 imágenes conservan los SHA256 originales. La colección integrada y los 15 archivos JSON individuales coinciden.
- Interfaz web: carga/ejecución de Unidad 2 con 12 épocas, 11 entradas y 248 muestras TE; informe de selección, métricas y gráficos visible.
- Exportación desde ficha a unit-2.json y reimportación a Personal verificadas desde la interfaz.
- Electron compilado desde file://: biblioteca e imágenes disponibles sin fetch HTTP; cDCGAN ejecutada con 9 nodos, 3 activos y 699 ventanas TE; selección, métricas y gráficos visibles. El diseño previo quedó archivado en Personal.
- Captura de biblioteca: docs/v0.5-library.png.
- Datos sintéticos usados para los nuevos ejemplos; esta verificación no demuestra rendimiento predictivo en datos reales ni compatibilidad nativa de Windows.

## Comprobaciones anteriores

- TypeScript estricto y compilación Vite/Electron: correctos.
- Suite unittest: 26 pruebas, con diez arquitecturas como subpruebas; correcta.
- Dependencias Python: `pip check` sin incompatibilidades.
- Pipeline desde la interfaz web: completado con datos sintéticos, 144 observaciones de prueba y early stopping en 7 epochs. Los cuatro paneles mostraron datos.
- Arranque de Electron compilado en macOS: ventana nativa y renderer local verificados.
- Sintaxis del lanzador Bash: correcta.
- Sin verificación nativa en Windows. Descargas yfinance verificadas para EURUSD y para portafolio SPY/QQQ/TLT (ver registros abajo). Instaladores de escritorio quedan fuera de esta base.

## Ampliación de indicadores y diseño de redes

- Fórmulas de VWAP, SMA, BB y MACD verificadas con datos sintéticos.
- Activar/desactivar indicadores y causalidad de características comprobados.
- Selección de 16 entradas y dimensiones 16×64, 64×32, 32×1 verificadas.
- Dropout, BatchNorm y activaciones en redes conectadas comprobados.
- Unión de ramas y detección de columnas contradictorias comprobadas.
- Ejemplo JSON ejecutado mediante HTTP, incluyendo ramificación de una arquitectura a dos entrenadores.
- Ejemplo cargado y entrenado desde la interfaz: 9 nodos, 16 entradas, 147 observaciones de prueba y 12 epochs. Seleccionar nodos conserva los gráficos; se verificaron los checkboxes de indicadores y de características.
- Versión correspondiente a la ampliación de redes: 0.3.0.

## Versiones Python instaladas

| Paquete | Versión |
| --- | --- |
| torch | 2.15.0 |
| pandas | 2.3.3 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| fastapi | 0.162.0 |
| uvicorn | 0.54.0 |
| arch | 8.0.0 |
| statsmodels | 0.15.0 |
| yfinance | 0.2.66 |

## Versiones principales JavaScript instaladas

| Paquete | Versión |
| --- | --- |
| Electron | 42.11.18 |
| React / React DOM | 19.3.1 |
| React Flow | 12.12.0 |
| TypeScript | 5.9.3 |
| Vite | 6.4.5 |
| Recharts | 3.10.1 |

Consultar `quant-blueprint-studio/package-lock.json` para el árbol npm completo.

## v0.3.0

- Catorce pruebas Python correctas, incluyendo progreso por época, diagnóstico de filas insuficientes y recuperación ampliando historia.
- Plot verificado para columnas, descomposición, pronósticos, pérdidas, equity y señales; valida fuentes incompatibles y series inexistentes.
- `npm run test:graph`: resumen 16→64→32→1, sincronización automática de entradas, bloqueo de discrepancias manuales e indicadores desactivados.

- Volumen cero reproducido con datos de prueba: VWAP informa cómo desactivarlo y el pipeline vuelve a tener muestras válidas al hacerlo.

- Diagnóstico sobre Yahoo real: EURUSD=X devolvió 259 observaciones en la consulta de un año, todas con volumen cero. La sesión del usuario con 10 años también produjo el mensaje específico de VWAP y mostró eventos por nodo desde Electron.
- Comprobación visual independiente: ejemplo de 11 nodos, 147 observaciones de prueba, 16 entradas coincidentes y dos gráficos Plot (precio/VWAP/BB y pérdidas de 12 épocas).
- La app de escritorio se reinició con v0.3.0. La versión aparece en su cabecera.

## v0.3.1

- `CumulativeLogReturn` verificado contra la suma acumulada de las filas útiles, incluida su primera observación.
- Columna comprobada en Plot y en la preparación de entradas.
- Quince pruebas Python, compilación y comprobaciones del grafo correctas. Unión de ramas con distinto calentamiento conserva una acumulación coherente sobre las fechas comunes.

## v0.4.0

- 24 pruebas Python: métricas comparadas con cálculos independientes, alpha/beta OLS conocidos, solución analítica de mínima varianza, restricciones de pesos y máximo Sharpe al menos igual al candidato de pesos iguales.
- Modificar únicamente precios del holdout no modifica los pesos de Markowitz.
- Costes, buy-and-hold, benchmark, fuentes de Plot, paleta y pipeline API de portafolio comprobados.
- Comprobaciones del grafo: pines nuevos, ejemplo de portafolio y presentación inmediata de títulos/colores/leyendas.

- Ejemplo visual ejecutado: tres activos sintéticos, 528 retornos de estimación y 227 de holdout, seis filas de métricas (activos más asignaciones), pesos y dos gráficos.
- Color Coral y título SPY / QQQ / TLT · Holdout aplicados sin repetir ejecución; se conservaron las métricas.
- Integración Yahoo real: SPY, QQQ y TLT de un año, 251 fechas comunes, benchmark SPY, optimización con límite de peso 60%, 75 observaciones holdout y seis filas de métricas.

## v0.4.1

- Métricas separadas de cartera, activos y benchmark comprobadas; retorno de cartera contrastado con composición diaria de los activos.
- Renderizado de las once tarjetas, selección personalizada por defecto y compatibilidad con resultados anteriores mediante `npm run test:portfolio-ui`.
- Comprobación visual: once métricas visibles, selector cambiado a mínima varianza, valores y pesos actualizados sin repetir ejecución. Compilación, 25 pruebas Python y pruebas de grafo/renderizado correctas.

## Auditoría de signos de alpha / Sharpe / Sortino

- Identidad OLS contrastada con beta por covarianza/varianza y alpha por diferencia de medias ajustada a beta.
- Caso de alpha positivo con ROI, Sharpe y Sortino negativos verificado con RF anual 0% y 5%.
- La auditoría de la configuración concreta se guarda únicamente en `quant-blueprint-studio/reports/local/` (excluido de Git). Sin cambios en las fórmulas de producción tras esta revisión.
