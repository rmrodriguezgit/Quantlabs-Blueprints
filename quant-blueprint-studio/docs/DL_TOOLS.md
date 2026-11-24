# DL Tools · herramientas de Deep Learning independientes

## Arquitectura y flujo de datos
```
Blueprint (UI) ──frame──► adaptador engine/nodes/dl_tool_nodes.py ──payload tipado──► registry.get(tool)(params)
                                                                         │  with tool:  load → warmup → execute → unload
frame + columnas nuevas + informe ◄──── ToolResult (outputs, inference_ms, device, weights) ◄┘
```
Las herramientas no se importan entre sí ni importan el motor o el UI (lo comprueba un test). Sólo se comunican por puertos tipados.

## Árbol
```
backend/dl_tools/
  core/contracts.py   Port, Capabilities, ToolSpec (NodeDescriptor), RunContext, ToolResult, ToolError, BaseDLTool
  core/registry.py    @register, discover(), get(), describe() → JSON
  core/device.py      auto → mps/cuda/cpu, fallback a cpu con aviso
  core/memory.py      release(): pesos a cpu, gc, empty_cache; device_bytes()
  core/weights.py     checkpoint → ajuste sobre la entrada → init sembrada; fit(); windows()
  core/graph.py       ejecutor mínimo: valida puertos, orden topológico, unload en finally
  tools/dl/tabular_forecast/    schema.py · model.py (MLP/1D-CNN/LSTM) · tool.py · config.json
  tools/dl/sequence_classifier/ schema.py · model.py (GRU + embedding) · tool.py · config.json
  tools/dl/anomaly_autoencoder/ schema.py · model.py (autoencoder) · tool.py · config.json
  checkpoints/        <tool>-<forma>-<weights_version>.pt (opcional)
  examples/run_blueprint.py
backend/tests/test_dl_tools.py
```

## Herramientas
| Tool | Entrada | Salida |
|---|---|---|
| `dl_tabular_forecast` | `series: vector` | `forecast`, `lower`, `upper`, `residuals: matrix`, `confidence` |
| `dl_sequence_classifier` | `series: vector`, `labels?` | `label`, `probabilities`, `latent`, `history_labels`, `accuracy` |
| `dl_anomaly_autoencoder` | `data: matrix` | `score`, `is_anomaly: mask`, `dimension_error`, `threshold`, `anomaly_rate` |

Pesos: si existe un checkpoint compatible se usa; si no, se registra un aviso y se ajusta (con semilla) sobre la parte inicial de la entrada; las métricas se calculan en el tramo final no visto.

## Ejemplo
```bash
.venv/bin/python -m backend.dl_tools.examples.run_blueprint
```
Encadena pronóstico → anomalías sobre sus residuos y un clasificador; detecta el choque inyectado y muestra `all_released: true` y 0 bytes asignados en MPS tras terminar.

En la app: área **12 / DL TOOLS** y el ejemplo «DL Tools · pronóstico, clasificación y anomalías (SPY)».

## Arquitectura de cada herramienta: una entrada, capas intermedias configurables, una salida
| Herramienta | Capa de entrada (única) | Capas intermedias | Capa de salida (única) |
|---|---|---|---|
| Pronóstico | MLP · 1D-CNN · LSTM (`architecture`, `input_units`) | 0–8 | Linear de `horizon` neuronas |
| Clasificador | GRU (`input_units`) | 0–8 (la última es el embedding latente) | Linear de `classes` neuronas |
| Autoencoder | Linear + activación | 0–8 en el codificador → cuello `latent` → espejo en el decodificador | Linear de nº de columnas |

Capas intermedias, de tres formas (de mayor a menor prioridad):
1. Conectar al pin **network** la cadena *Network Input → Network Layer… → Network Output* (Dense crea capa; Dropout, BatchNorm o una activación modifican la Dense anterior).
2. «Capas a medida»: `64:relu:0.1:bn, 32:tanh` (unidades:activación:dropout:bn).
3. `hidden_layers` × `units` con `activation`, `dropout` y `batch_norm` iguales.

Optimizadores: adam, adamw, sgd (momentum), rmsprop, adagrad; con `lr`, `weight_decay`, `batch_size`, `epochs` máximas y `patience`.
El 15 % final del tramo de ajuste es validación interna: se restauran los pesos de la mejor época y se para pronto si no mejora.

## Comprobación de que cumplen su función (tests `DoesItsJob`)
- Pronóstico: en un AR(0.7) llega al óptimo teórico (RMSE ≈ ruido) y mejora a la mejor línea base.
- Clasificador: en una secuencia predecible supera a la clase mayoritaria en > 10 puntos.
- Autoencoder: detecta ≥ 8 de 10 filas donde se rompe una correlación sin cambiar las marginales.

## DL · LSTM vs GRU (secuencias) — réplica de «Cómo programar: LSTM y GRU»
Herramienta `dl_recurrent_compare` (`tools/dl/recurrent_compare/`) y ejemplo «Cómo programar: LSTM y GRU · siguiente intradía (SPY)».

| Segmento | Dónde |
|---|---|
| 01 Preparar secuencias (batch × 20 × 11) | `tool.make_sequences` |
| 02 LSTM · 03 GRU · 04 bidireccional | `model.RecurrentRegressor.input_layer` |
| 05 Estado final (`state[0]` si LSTM) · 06 `cat(h[-2], h[-1])` | `RecurrentRegressor.summary` |
| 07 Cabeza `nn.Linear(64 si bi, si no 32, 1)` | `RecurrentRegressor.output_layer` (capas intermedias opcionales) |
| 08 Sin arrastrar estados | cada `forward` empieza con memoria nueva (test) |
| 09 Selección por `validation_MSE` | tabla de comparación con parámetros |
| 10 Ridge + `inverse_transform` a bps | métricas TE y estimación de la próxima sesión |

Los tests `backend/tests/test_dl_recurrent.py` comprueban cada segmento.
