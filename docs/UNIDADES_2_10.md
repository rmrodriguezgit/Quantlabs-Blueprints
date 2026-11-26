# Unidades 2–10 · Depuración, catálogo de herramientas y Blueprints

## 1 · Monolitos depurados

| Módulo | Problema | Acción |
| --- | --- | --- |
| `study` (`backend/engine/nodes/study_nodes.py`) | Un nodo con `topic` hacía ventanas, escalado, arquitectura, entrenamiento, selección, evaluación y backtest para U2–U10 | Sustituido en los ejemplos por herramientas atómicas. Queda como **«Learning Experiment (legado)»** (categoría 99 / LEGADO) sólo para abrir grafos personales antiguos; la validación avisa al usarlo. |
| `rl_agent` | Envoltorio de `study(topic='double_dqn')` | Sustituido en U10 por `rl_*`. Sigue en la receta `research-rl-agent` (aviso de legado). |
| `unit-8.json`, `unit-9.json` | Ejemplos monolíticos duplicados de `unit-8-tools` / `unit-9-tools` | Retirados de la biblioteca. |
| `unit-2…7.json`, `unit-9-cdcgan.json`, `unit-10.json` | `source → features → study → plot` | Reescritos con una herramienta por segmento (mismo id e infografía). |

Los scripts de la raíz (`ae_tools.py`, `gen_tools.py`, `autoencoders_unidad8.py`, `generativos_unidad9.py`) se auditaron: ya son funciones separadas + un script conductor; no se modificaron.

## 2 · Inventario por unidad

Todas las infografías están en `quant-blueprint-studio/public/infographics/`. **Falta infografía propia para ninguna unidad**; U9 tiene dos (CVAE/cGAN y cDCGAN).

| Unidad | Infografía | Ejemplo → Blueprint | Herramientas (segmento) |
| --- | --- | --- | --- |
| U2 Feedforward | unidad-2.png | `unit-2` | source (01), features (02), sup_target (03), sup_split (04), sup_scale (05), sup_mlp (06), sup_loss (07), sup_train (08–09), sup_evaluate (10) |
| U3 Entrenamiento | unidad-3.png | `unit-3` | sup_target/split/scale (01), sup_init (02), sup_optimizer ×3 (03), sup_loss (05), sup_train (04–08), sup_select (09), sup_evaluate (10) |
| U4 Regularización | unidad-4.png | `unit-4` | sup_target dirección (01), sup_init + sup_mlp (02), sup_regularizer ×5 (03–07), sup_loss BCE + sup_train (08), sup_select (09), sup_evaluate (10) |
| U5 Hiperparámetros | unidad-5.png | `unit-5` | sup_split (01), sup_hpo (02–08), sup_scale + sup_train (09), sup_evaluate (10) |
| U6 CNN 1D | unidad-6.png | `unit-6` | sup_scale (01), sup_windows (02/04), sup_split (03), sup_cnn (05–07), sup_mlp aplanada (08), sup_train + sup_select (09), sup_evaluate (10) |
| U7 LSTM/GRU | unidad-7.png | `unit-7` | sup_windows (01), sup_rnn ×4 (02–07), sup_train (08), sup_select (09), sup_evaluate (10) |
| U8 Autoencoders | unidad-8.png | `unit-8-tools` (existente) | ae_split … ae_test (01–10) |
| U9 Generativos | unidad-9.png | `unit-9-tools` (existente) | gen_windows … gen_evaluate (01–10) |
| U9 ampliación | unidad-9-cdcgan.png | `unit-9-cdcgan` | gen_windows (01), gen_condition (02), gen_split + gen_scale (03), **gen_dcgan_train** (04–08), gen_select + **gen_regime_compare** (09), gen_evaluate (10) |
| U10 Double DQN | unidad-10.png | `unit-10` | sup_target/split/scale (01), rl_env (02–03), rl_qnet (04), rl_replay (05–06), rl_target (07), rl_train (08), rl_sync (09), rl_baselines + rl_evaluate (10) |

Herramientas nuevas: 15 `sup_*` (`backend/engine/nodes/sup_pipeline_nodes.py`), 8 `rl_*` (`rl_pipeline_nodes.py`), 2 `gen_*` (en `gen_pipeline_nodes.py`). Contrato común en `pipeline_state.py` (no es herramienta).

## 3 · Reglas de diseño

- Cada herramienta: `(frame, params, progress) → (frame, informe)`, errores explícitos en español, sin importar otras familias de herramientas (lo comprueba un test AST).
- La conexión ocurre sólo por el pin `frame` del grafo; los arrays/redes viajan en un registro en proceso referenciado desde `frame.attrs`.
- Sin fuga: objetivo t+1, escaladores/pesos/Ridge sólo con TR, VA elige checkpoint/modelo, TE se mide una vez.

## 4 · Verificación

- `backend/tests/test_unit_blueprints.py`: (a) U2…U10 presentes sin huecos, (b) cada segmento tiene su herramienta en el Blueprint, (c) ningún ejemplo `unit-*` usa `study`/`rl_agent`, (d) cada Blueprint se ejecuta de punta a punta (épocas reducidas) y cada herramienta devuelve informe JSON válido. **4/4 OK.**
- `npm run typecheck` y `test:graph/library/catalog/connections/validation/scaling-ui/portfolio-ui`: OK (contadores de biblioteca 41→39 y 11 infografías referenciadas).
- Regenerar los Blueprints: `python scripts/build_unit_blueprints.py`.

## Pendiente

- cDCGAN multiactivo (embedding de activo, pines asset_b/asset_c del monolito) y exportación de pesos/escalas a archivo: no incluidos en el pipeline modular.
- No se verificó la interfaz Electron visualmente ni ejecuciones en Colab/T4.
