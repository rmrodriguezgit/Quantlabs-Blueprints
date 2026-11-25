"""Genera el ejemplo multicapa del submenú «17 / ML · REDES NEURONALES · TIME SERIES & YFINANCE».

Uso:  python scripts/build_nn_blueprints.py
Escribe public/blueprints/nn-multicapa-yfinance.json y añade/actualiza la entrada en ambas library.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BP = ROOT / 'public/blueprints'
EID = 'nn-multicapa-yfinance'

INPUT = {'mode': 'lags', 'features': 'LogReturn,Range,LogVolume', 'target': 'LogReturn', 'horizon': 1, 'lags': 10, 'lag_stride': 1,
         'nan_policy': 'drop', 'activation': 'linear', 'projection_units': 0, 'optimizer': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001, 'clip': 1.0}
DATA = {'source': 'yfinance', 'interval': '1d', 'range_mode': 'period', 'period': '10y', 'price_column': 'close', 'retries': 3, 'fallback': 'synthetic'}
EMB = {'weekday': True, 'month': True, 'regime': True, 'ticker': False, 'dim': 4, 'fusion': 'concat'}

BASE = [  # (id, kind, label, params)
    ('spy', 'nn_data', '1 · Datos yfinance · SPY', {**DATA, 'ticker': 'SPY'}),
    ('guard', 'nn_guard', '2 · Guardia de errores', {'policy': 'repair'}),
    ('input', 'nn_input', '3 · Capa de entrada · 3 rasgos × 10 lags', INPUT),
    ('split', 'nn_split', '4 · TR / VA / TE (purga h)', {'train_ratio': 0.7, 'val_ratio': 0.15, 'mode': 'holdout'}),
    ('scaler', 'nn_scaler', '5 · Escalador Standard (fit TR)', {'method': 'standard', 'operation': 'fit_transform', 'scale_target': True, 'persist': True, 'save_as': 'spy_scaler'}),
    ('emb', 'nn_embedding', '6 · Embeddings · día, mes, régimen', EMB),
    ('h1', 'nn_hidden', '7a · Oculta 128 · GELU · LayerNorm', {'units': '128', 'activation': 'gelu', 'norm': 'layer', 'dropout': 0.1, 'name': 'Oculta A (128)'}),
    ('h2', 'nn_hidden', '7b · Oculta 64 · residual · L2', {'units': '64', 'activation': 'inherit', 'norm': 'layer', 'dropout': 0.1, 'l2': 1e-5, 'residual': True, 'name': 'Oculta B (64)'}),
    ('h3', 'nn_hidden', '7c · Oculta 32 · SiLU · lr propio', {'units': '32', 'activation': 'silu', 'dropout': 0.05, 'optimizer': 'adam', 'lr': 0.0005, 'name': 'Oculta C (32)'}),
    ('out', 'nn_output', '8 · Capa de salida · regresión t+1', {'task': 'regression', 'activation': 'linear', 'loss': 'huber', 'huber_delta': 1.0}),
    ('train', 'nn_train', '9 · Entrenar · checkpoint VA', {'epochs': 60, 'batch_size': 64, 'patience': 10, 'scheduler': 'plateau', 'save_as': 'spy_mlp_base'}),
    ('pred', 'nn_predict', '11 · Predicción SPY · banda 90%', {'interval': 0.9, 'export': 'spy_predicciones'}),
]
TUNE = [
    ('qqq', 'nn_data', '1 · Datos yfinance · QQQ (reciente)', {**DATA, 'ticker': 'QQQ', 'period': '5y'}),
    ('guard_q', 'nn_guard', '2 · Guardia de errores', {'policy': 'repair'}),
    ('input_q', 'nn_input', '3 · Capa de entrada (idéntica)', INPUT),
    ('split_q', 'nn_split', '4 · TR / VA / TE', {'train_ratio': 0.7, 'val_ratio': 0.15}),
    ('scaler_q', 'nn_scaler', '5 · Escalador Standard (fit TR QQQ)', {'method': 'standard', 'operation': 'fit_transform'}),
    ('emb_q', 'nn_embedding', '6 · Embeddings (mismos campos)', EMB),
    ('ft', 'nn_finetune', '10 · Ajuste fino SPY → QQQ', {'source': 'pin', 'freeze': 'hidden_bottom', 'freeze_layers': 1, 'lr_factor': 0.1, 'epochs': 30, 'patience': 6, 'save_as': 'qqq_mlp_ft'}),
    ('pred_q', 'nn_predict', '11 · Predicción QQQ · en vivo', {'interval': 0.9}),
]


def graph() -> dict[str, Any]:
    nodes, edges = [], []
    for row, chain in enumerate((BASE, TUNE)):
        for i, (nid, kind, label, params) in enumerate(chain):
            col = i if row == 0 else (i if i < 6 else i + 4)
            nodes.append({'id': nid, 'type': 'blueprint', 'position': {'x': 40 + 290 * col, 'y': 40 + 300 * row},
                          'data': {'kind': kind, 'label': label, 'params': params}})
        for a, b in zip(chain, chain[1:]):
            edges.append({'id': f'{a[0]}-{b[0]}', 'source': a[0], 'target': b[0], 'sourceHandle': 'frame', 'targetHandle': 'frame', 'animated': True})
    edges.append({'id': 'train-ft', 'source': 'train', 'target': 'ft', 'sourceHandle': 'frame', 'targetHandle': 'pretrained', 'animated': True})
    return {'nodes': nodes, 'edges': edges}


ENTRY = {
    'id': EID,
    'title': 'Redes Neuronales · MLP multicapa 128 → 64 → 32 (yfinance, embeddings, ajuste fino)',
    'area': 'ML / Redes Neuronales · Time Series & yFinance',
    'description': 'Submenú 17 de extremo a extremo: yfinance con fallback, guardia de errores, entrada con lags, TR/VA/TE purgado, escalador '
                   'persistido, embeddings de calendario y régimen, tres capas ocultas duplicadas con configuración propia, salida Huber, '
                   'entrenamiento con checkpoint, ajuste fino SPY → QQQ y predicción en vivo con bandas.',
    'instructions': [
        'Propósito: pronosticar el log-retorno del día siguiente con una MLP multicapa y reutilizarla en otro activo por ajuste fino.',
        'Fila 1 (modelo base, SPY): Datos → Guardia → Entrada (3 rasgos × 10 lags) → TR/VA/TE → Escalador → Embeddings → Oculta 128 → 64 → 32 → Salida → Entrenar → Predicción.',
        'Fila 2 (ajuste fino, QQQ): misma entrada y embeddings; «Ajuste fino» recibe el modelo base por el pin pretrained, congela la entrada y la primera oculta y entrena con lr × 0.1.',
        'Sin internet: fallback = synthetic genera una serie sintética (avisado en la bitácora). Pon fallback = none para exigir datos reales.',
        'Duplicar capas: copia/pega una «Capa oculta» y encadénala, o usa «Repeticiones». activation = inherit hereda la de la capa anterior.',
        'Resultado: pérdidas TR/VA, RMSE/MAE/R²/acierto direccional en TE frente al pronóstico ingenuo 0, cobertura de la banda 90 % y la predicción en vivo de las últimas barras.',
        'Artefactos: artifacts/nn/scalers/spy_scaler.joblib, artifacts/nn/models/spy_mlp_base.pt, qqq_mlp_ft.pt y artifacts/nn/exports/spy_predicciones.csv.',
    ],
    'graph': graph(),
}


def main() -> None:
    (BP / f'{EID}.json').write_text(json.dumps({'format': 'quantlabs-blueprint-library', 'schemaVersion': 1, 'entries': [ENTRY]}, ensure_ascii=False, indent=2) + '\n')
    for lib in (BP / 'library.json', ROOT / 'src/renderer/data/library.json'):
        data = json.loads(lib.read_text())
        data['entries'] = [e for e in data['entries'] if e['id'] != EID] + [ENTRY]
        lib.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f'{EID}: {len(ENTRY["graph"]["nodes"])} nodos, {len(ENTRY["graph"]["edges"])} conexiones')


if __name__ == '__main__':
    main()
