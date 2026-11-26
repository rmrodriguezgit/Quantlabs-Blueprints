"""Genera un Blueprint autocontenido por ejemplo de las Unidades 2–10 con herramientas atómicas.

Uso:  python scripts/build_unit_blueprints.py
Escribe public/blueprints/unit-*.json y sincroniza las entradas en public/blueprints/library.json y
src/renderer/data/library.json. Los Blueprints «unit-8-tools» y «unit-9-tools» ya existían y se conservan.
Los Blueprints monolíticos (nodo ``study``) de las Unidades 8 y 9 se retiran porque duplicaban esos ejemplos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BP = ROOT / 'public/blueprints'
FEATURES = {'lags': 0, 'use_rsi': True, 'use_ema': True, 'use_sma': True, 'use_macd': True, 'use_vwap': True, 'use_bb': True}
DATA = [('data', 'source', 'SPY · synthetic / yfinance', {'source': 'synthetic', 'ticker': 'SPY', 'years': 6}),
        ('features', 'features', '11 indicadores causales', FEATURES)]
RETIRED = ('unit-8', 'unit-9')  # monolitos duplicados por unit-8-tools / unit-9-tools


def graph(chain: list[tuple[str, str, str, dict[str, Any]]]) -> dict[str, Any]:
    nodes = [{'id': nid, 'type': 'blueprint', 'position': {'x': 40 + 300 * (i % 6), 'y': 40 + 220 * (i // 6)},
              'data': {'kind': kind, 'label': label, 'params': params}} for i, (nid, kind, label, params) in enumerate(chain)]
    edges = [{'id': f'{a[0]}-{b[0]}', 'source': a[0], 'target': b[0], 'sourceHandle': 'frame', 'targetHandle': 'frame', 'animated': True}
             for a, b in zip(chain, chain[1:])]
    return {'nodes': nodes, 'edges': edges}


def guide(purpose: str, chain: list[tuple[str, str, str, dict[str, Any]]], result: str) -> list[str]:
    steps = ' → '.join(label for _, _, label, _ in chain)
    return [f'Propósito: {purpose}',
            'Datos de prueba: serie sintética reproducible (6 años, sin internet). Cambia SPY a yfinance para datos reales.',
            f'Cadena: {steps}. Cada herramienta recibe el pin frame del paso anterior y entrega el suyo al siguiente.',
            f'Resultado esperado: {result}',
            'Pulsa Execute Blueprint. Cada herramienta publica su propio informe (tabla, métricas y gráficos) debajo del lienzo.']


SUP_HEAD = [('target', 'sup_target', '03 · Definir el objetivo', {}), ('split', 'sup_split', '04 · Separar el tiempo', {}),
            ('scale', 'sup_scale', '05 · Escalar sólo con train', {})]

EXAMPLES: list[dict[str, Any]] = [
    {'id': 'unit-2', 'title': 'Unidad 2 · Feedforward 11 → 64 → 32 → 1', 'area': 'Redes feedforward', 'infographic': 'unidad-2.png',
     'description': 'Diez segmentos de la infografía, una herramienta por paso: objetivo intradía, cortes, escalado TR, MLP, pérdida, checkpoint y evaluación con Ridge y costes.',
     'purpose': 'estimar el intradía de la sesión siguiente (bps) con una MLP 11 → 64 → 32 → 1.',
     'result': 'RMSE/MAE/R² en test frente a Ridge, curva de pérdidas del checkpoint y resultado neto de costes.',
     'chain': [*DATA, *SUP_HEAD, ('mlp', 'sup_mlp', '06 · Construir la MLP', {'widths': '64,32', 'name': 'MLP 11→64→32→1'}),
               ('loss', 'sup_loss', '07 · Elegir la pérdida', {'loss': 'MSE'}),
               ('train', 'sup_train', '08–09 · Actualizar pesos y guardar checkpoint', {'epochs': 30, 'patience': 8}),
               ('evaluate', 'sup_evaluate', '10 · Evaluar y comparar', {})]},
    {'id': 'unit-3', 'title': 'Unidad 3 · SGD / RMSprop / AdamW', 'area': 'Entrenamiento', 'infographic': 'unidad-3.png',
     'description': 'Misma MLP, mismos pesos iniciales y mismo presupuesto; sólo cambia el optimizador.',
     'purpose': 'comparar tres reglas de actualización (SGD, RMSprop, AdamW) en igualdad de condiciones.',
     'result': 'tabla validation_MSE por optimizador, modelo seleccionado sin mirar test y su evaluación frente a Ridge.',
     'chain': [*DATA, ('target', 'sup_target', '01 · Objetivo y_bps', {}), ('split', 'sup_split', '01 · Separar por tiempo', {}),
               ('scale', 'sup_scale', '01 · x_scaler / y_scaler', {}),
               ('init', 'sup_init', '02 · Fijar la inicialización', {'scheme': 'kaiming_normal'}),
               ('mlp', 'sup_mlp', 'Plantilla MLP 64 → 32', {'widths': '64,32', 'include_as_candidate': False}),
               ('sgd', 'sup_optimizer', '03 · SGD', {'optimizer': 'SGD', 'lr': 0.01}),
               ('rmsprop', 'sup_optimizer', '03 · RMSprop', {'optimizer': 'RMSprop', 'lr': 0.001}),
               ('adamw', 'sup_optimizer', '03 · AdamW', {'optimizer': 'AdamW', 'lr': 0.001}),
               ('loss', 'sup_loss', '05 · Calcular salida y pérdida', {'loss': 'MSE'}),
               ('train', 'sup_train', '04–08 · Entrenar con igual presupuesto', {'epochs': 30, 'patience': 0}),
               ('select', 'sup_select', '09 · Seleccionar con validación', {}),
               ('evaluate', 'sup_evaluate', '10 · Evaluar fuera de muestra', {})]},
    {'id': 'unit-4', 'title': 'Unidad 4 · Regularización y dirección', 'area': 'Regularización', 'infographic': 'unidad-4.png',
     'description': 'Etiqueta de dirección, referencia común y una herramienta por técnica: Dropout, L1, L2, BatchNorm y early stopping.',
     'purpose': 'estimar la probabilidad de alza de la sesión siguiente y ver qué regularización generaliza mejor.',
     'result': 'BCE de validación por variante, variante seleccionada y probabilidades en test con umbral 0.5.',
     'chain': [*DATA, ('target', 'sup_target', '01 · Definir las etiquetas', {'task': 'direction'}),
               ('split', 'sup_split', 'Separar el tiempo', {}), ('scale', 'sup_scale', 'Escalar sólo con train', {}),
               ('init', 'sup_init', '02 · Crear una referencia común', {'scheme': 'kaiming_normal'}),
               ('mlp', 'sup_mlp', '02 · Baseline (misma MLP)', {'widths': '64,32', 'name': 'Baseline'}),
               ('dropout', 'sup_regularizer', '03 · Aplicar Dropout', {'kind': 'Dropout', 'value': 0.3}),
               ('l1', 'sup_regularizer', '04 · Añadir L1', {'kind': 'L1', 'value': 0.00001}),
               ('l2', 'sup_regularizer', '05 · Añadir L2', {'kind': 'L2', 'value': 0.001}),
               ('bn', 'sup_regularizer', '06 · Evaluar BatchNorm', {'kind': 'BatchNorm', 'value': 0}),
               ('early', 'sup_regularizer', '07 · Aplicar Early Stopping', {'kind': 'EarlyStopping', 'value': 5}),
               ('loss', 'sup_loss', 'BCE con logits', {'loss': 'BCE'}),
               ('train', 'sup_train', '08 · Comparar pérdidas equivalentes', {'epochs': 30, 'patience': 0, 'restore_best': False}),
               ('select', 'sup_select', '09 · Seleccionar la variante', {}),
               ('evaluate', 'sup_evaluate', '10 · Evaluar probabilidades y costes', {'threshold': 0.5})]},
    {'id': 'unit-5', 'title': 'Unidad 5 · Búsqueda temporal', 'area': 'Hiperparámetros', 'infographic': 'unidad-5.png',
     'description': 'Búsqueda aleatoria con folds causales dentro de train, escaladores por fold y modelo final con checkpoint en validación.',
     'purpose': 'optimizar una MLP (anchura, profundidad, lr, weight decay) sin tocar validación ni test durante la búsqueda.',
     'result': 'tabla de pruebas con RMSE medio por fold, configuración ganadora reentrenada y evaluada una vez en test.',
     'chain': [*DATA, ('target', 'sup_target', 'Objetivo y_bps', {}), ('split', 'sup_split', '01 · Separar desarrollo y test', {}),
               ('hpo', 'sup_hpo', '02–08 · Espacio, folds y selección', {'trials': 6}),
               ('scale', 'sup_scale', '09 · Escalar con train completo', {}), ('loss', 'sup_loss', 'Pérdida MSE', {'loss': 'MSE'}),
               ('train', 'sup_train', '09 · Ajustar el modelo final', {'epochs': 30, 'patience': 8}),
               ('evaluate', 'sup_evaluate', '10 · Evaluar y documentar', {})]},
    {'id': 'unit-6', 'title': 'Unidad 6 · CNN 1D', 'area': 'Redes convolucionales', 'infographic': 'unidad-6.png',
     'description': 'Ventanas de 20 sesiones × 11 indicadores, Conv1d 32 → 64, pooling global y MLP aplanada de referencia.',
     'purpose': 'usar veinte sesiones de once indicadores para estimar el intradía siguiente con filtros compartidos.',
     'result': 'validation_MSE de CNN_1D frente a la MLP aplanada, modelo seleccionado y predicción desescalada en test.',
     'chain': [*DATA, ('target', 'sup_target', 'Objetivo y_bps', {}), ('split', 'sup_split', '03 · Asignar por fechas', {}),
               ('scale', 'sup_scale', '01 · Preparar indicadores', {}), ('windows', 'sup_windows', '02/04 · Ventanas por activo', {'lookback': 20}),
               ('cnn', 'sup_cnn', '05–07 · Conv1d → Conv1d → pool', {'channels': '32,64', 'kernel': 3, 'name': 'CNN_1D'}),
               ('mlp', 'sup_mlp', '08 · MLP de referencia', {'widths': '64', 'name': 'MLP aplanada'}),
               ('loss', 'sup_loss', 'Pérdida MSE', {'loss': 'MSE'}), ('train', 'sup_train', '09 · Entrenar', {'epochs': 25, 'patience': 6}),
               ('select', 'sup_select', '09 · Seleccionar', {}), ('evaluate', 'sup_evaluate', '10 · Evaluar la predicción', {})]},
    {'id': 'unit-7', 'title': 'Unidad 7 · LSTM / GRU / bidireccional', 'area': 'Redes recurrentes', 'infographic': 'unidad-7.png',
     'description': 'Secuencias (batch, 20, 11), cuatro celdas recurrentes como herramientas independientes, selección por validación y Ridge.',
     'purpose': 'estimar el siguiente intradía a partir de veinte sesiones de indicadores con memoria recurrente.',
     'result': 'validation_MSE y parámetros de LSTM, GRU y sus versiones bidireccionales, y evaluación en test frente a Ridge.',
     'chain': [*DATA, ('target', 'sup_target', 'Objetivo y_bps', {}), ('split', 'sup_split', 'Separar el tiempo', {}),
               ('scale', 'sup_scale', 'Escalar sólo con train', {}), ('windows', 'sup_windows', '01 · Preparar secuencias', {'lookback': 20}),
               ('lstm', 'sup_rnn', '02 · Crear una LSTM', {'cell': 'LSTM', 'hidden': 32}),
               ('gru', 'sup_rnn', '03 · Crear una GRU', {'cell': 'GRU', 'hidden': 32}),
               ('bilstm', 'sup_rnn', '04 · LSTM bidireccional', {'cell': 'LSTM', 'hidden': 32, 'bidirectional': True}),
               ('bigru', 'sup_rnn', '04 · GRU bidireccional', {'cell': 'GRU', 'hidden': 32, 'bidirectional': True}),
               ('loss', 'sup_loss', 'Pérdida MSE', {'loss': 'MSE'}), ('train', 'sup_train', '08 · Entrenar sin arrastrar estados', {'epochs': 20, 'patience': 6}),
               ('select', 'sup_select', '09 · Seleccionar por validación', {}), ('evaluate', 'sup_evaluate', '10 · Evaluar y comparar con Ridge', {})]},
    {'id': 'unit-9-cdcgan', 'title': 'Unidad 9 ampliación · cDCGAN 1D en herramientas', 'area': 'Modelos generativos', 'infographic': 'unidad-9-cdcgan.png',
     'description': 'Ventanas de retornos, condición causal (volatilidad + momentum), embedding de régimen, generador ConvTranspose1d y discriminador Conv1d.',
     'purpose': 'generar escenarios de retornos condicionados por régimen bajista / lateral / alcista y compararlos con el mismo ruido.',
     'result': 'selección por ruido fijo en VA, trayectorias por régimen con el mismo z y métricas de colas en test.',
     'chain': [*DATA, ('windows', 'gen_windows', '01 · Preparar retornos', {'horizon': 16}),
               ('condition', 'gen_condition', '02 · Condiciones causales', {'use_log_vol': True, 'use_momentum': True}),
               ('split', 'gen_split', '03 · Separar', {}), ('scale', 'gen_scale', '03 · Escalar con train', {}),
               ('dcgan', 'gen_dcgan_train', '04–08 · Embeddings, G, D y alternancia', {'epochs': 40}),
               ('select', 'gen_select', '09 · Seleccionar con validación', {}),
               ('regimes', 'gen_regime_compare', '09 · Mismo ruido, tres regímenes', {}),
               ('evaluate', 'gen_evaluate', '10 · Evaluar colas y diversidad', {})]},
    {'id': 'unit-10', 'title': 'Unidad 10 · Double DQN', 'area': 'Aprendizaje por refuerzo', 'infographic': 'unidad-10.png',
     'description': 'Entorno de trading simulado, redes online/target, replay buffer, objetivo Double DQN, sincronización y evaluación greedy.',
     'purpose': 'aprender una política corto / efectivo / largo con costes y compararla con Q-learning, siempre largo y efectivo.',
     'result': 'checkpoint elegido por ROI neto en validación y curvas de test greedy con ROI, Sharpe, drawdown y rotación.',
     'chain': [*DATA, ('target', 'sup_target', '01 · Preparar datos (retornos t+1)', {}), ('split', 'sup_split', '01 · Train / validación / test', {}),
               ('scale', 'sup_scale', '01 · Escalar sólo con train', {'scale_target': False}),
               ('env', 'rl_env', '02–03 · Estado, acciones y entorno', {'cost_bps': 5}), ('qnet', 'rl_qnet', '04 · Crear las dos redes', {}),
               ('replay', 'rl_replay', '05–06 · Explorar y recordar', {}), ('bellman', 'rl_target', '07 · Objetivo Double DQN', {}),
               ('sync', 'rl_sync', '09 · Sincronizar y seleccionar', {}), ('train', 'rl_train', '08 · Actualizar online', {'epochs': 12}),
               ('baselines', 'rl_baselines', '10 · Referencias', {}), ('evaluate', 'rl_evaluate', '10 · Evaluar sin aprender', {})]},
]


def entry(ex: dict[str, Any]) -> dict[str, Any]:
    return {'id': ex['id'], 'title': ex['title'], 'area': ex['area'], 'description': ex['description'],
            'instructions': guide(ex['purpose'], ex['chain'], ex['result']), 'infographic': ex['infographic'], 'graph': graph(ex['chain'])}


def main() -> None:
    built = {ex['id']: entry(ex) for ex in EXAMPLES}
    for eid, e in built.items():
        (BP / f'{eid}.json').write_text(json.dumps({'format': 'quantlabs-blueprint-library', 'schemaVersion': 1, 'entries': [e]}, ensure_ascii=False, indent=2) + '\n')
    for eid in RETIRED:
        (BP / f'{eid}.json').unlink(missing_ok=True)
    for lib in (BP / 'library.json', ROOT / 'src/renderer/data/library.json'):
        data = json.loads(lib.read_text())
        data['entries'] = [built.get(e['id'], e) for e in data['entries'] if e['id'] not in RETIRED]
        lib.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(built)} Blueprints escritos; retirados: {", ".join(RETIRED)}')


if __name__ == '__main__':
    main()
