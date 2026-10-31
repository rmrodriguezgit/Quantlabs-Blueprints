# Verificación del proyecto (v0.4.0)

Fecha: 31 de octubre de 2026. Entorno local macOS, Python 3.12.12, Node.js 22.22.3, npm 10.9.8.

## Comprobaciones

- TypeScript estricto y compilación Vite/Electron: correctos.
- Suite unittest: 24 pruebas, con diez arquitecturas como subpruebas; correcta.
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
