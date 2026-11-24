# ML Pipeline · 8 módulos independientes

Cada módulo recibe y devuelve un `frame`, se configura en el Inspector y publica su propio informe.

| # | Módulo | Opciones clave |
|---|---|---|
| 1 | Seleccionar columnas | keep/drop, columnas, conservar siempre, sólo numéricas, renombrar |
| 2 | Rendimientos | simple `Pₜ/Pₜ₋ₖ−1` / log `ln(Pₜ/Pₜ₋ₖ)` / both, k, prefijos, sustituir |
| 3 | Limpieza de datos | duplicados, ±inf, faltantes (ffill, drop, mean…), atípicos (clip z/cuantil, remove z/IQR), constantes |
| 4 | Dividir TR/VA/TE | objetivo next/direction/same, horizonte, tarea, cronológico/aleatorio, fracciones, hueco, escalado en TR |
| 5 | Entrenar modelo ML | 10 regresores / 7 clasificadores o `Compare` (elige en VA), hiperparámetros, reentreno TR+VA |
| 6 | Evaluar el modelo | métricas TR/VA/TE frente a línea base, confusión, calibración, residuos |
| 7 | Realizar predicciones | TE y filas FUT (pronóstico), umbral, intervalo, señal ±1 |
| 8 | Visualizar resultados | real vs predicho, dispersión, residuos, estrategia ilustrativa, importancia, comparación |

Sin fuga: imputación, atípicos y escalado se ajustan con datos iniciales/TR; VA sólo elige; TE se mide una vez.
