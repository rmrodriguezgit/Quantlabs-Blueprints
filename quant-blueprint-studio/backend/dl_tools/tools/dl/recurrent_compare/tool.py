"""dl_recurrent_compare: the ten segments of «Cómo programar: LSTM y GRU» as one independent tool.

01 sequences (batch × lookback × F) · 02 LSTM · 03 GRU · 04 bidirectional · 05 final state · 06 concat
07 linear head → scalar return · 08 no state carried between windows · 09 select by validation MSE
10 evaluate on test, compare with Ridge, inverse-transform to the target units (bps).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ....core.contracts import BaseDLTool, Capabilities, Port, RunContext, ToolSpec
from ....core.registry import register
from ....core.weights import fit, load_config, seed_all
from .model import RecurrentRegressor
from .schema import RecurrentInput, RecurrentParams

CFG = load_config(Path(__file__).parent)


def make_sequences(x: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment 01: window ending at session t → (lookback, F); its target is y[t] (the next session's value)."""
    ends = np.arange(lookback - 1, len(x))
    sx = np.stack([x[e - lookback + 1:e + 1] for e in ends]).astype(np.float32)
    return sx, y[ends].astype(np.float32), ends


@register
class RecurrentCompareTool(BaseDLTool[RecurrentInput, RecurrentParams]):
    Input = RecurrentInput
    Params = RecurrentParams
    spec = ToolSpec(
        name=CFG['name'], version=CFG['version'], title='DL · LSTM vs GRU (secuencias)',
        description='Secuencias de indicadores → LSTM / GRU / bidireccional → selección por MSE de validación → test frente a Ridge en bps.',
        inputs=(Port(name='features', type='matrix', description='Indicadores n × F'),
                Port(name='target', type='vector', description='Valor de la sesión siguiente')),
        outputs=(Port(name='prediction', type='vector', description='Predicción en TE (unidades del objetivo × escala)'),
                 Port(name='actual', type='vector'), Port(name='next', type='scalar', description='Estimación para la próxima sesión'),
                 Port(name='comparison', type='table'), Port(name='selected', type='label')),
        params_schema=RecurrentParams.model_json_schema(),
        capabilities=Capabilities(est_ram_mb=CFG['est_ram_mb'], est_vram_mb=CFG['est_vram_mb'],
                                  expected_latency_ms=CFG['expected_latency_ms'], weights_version=CFG['weights_version']),
    )

    def _build(self, context: RunContext) -> tuple[Any, Literal['checkpoint', 'fitted_on_input', 'mock_init']]:
        return nn.ModuleDict(), 'mock_init'          # one sub-model per candidate, created in _infer

    def _infer(self, data: RecurrentInput, context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]:
        from sklearn.linear_model import Ridge
        p = self.params
        x_all = np.asarray(data.features, dtype=np.float64)
        y_all = np.asarray(data.target, dtype=np.float64) * p.target_scale
        sx, sy, ends = make_sequences(x_all, y_all, p.lookback)
        known = np.isfinite(sy)
        idx = np.flatnonzero(known)
        n = len(idx)
        if n < 60:
            raise ValueError(f'Sólo {n} secuencias con objetivo conocido; se necesitan ≥ 60.')
        n_tr, n_va = int(n * p.train_ratio), int(n * p.val_ratio)
        parts = {'TR': idx[:n_tr], 'VA': idx[n_tr:n_tr + n_va], 'TE': idx[n_tr + n_va:]}
        if min(len(v) for v in parts.values()) < 10:
            raise ValueError('Algún conjunto tiene < 10 secuencias: alarga la historia o cambia las fracciones.')
        # scalers fitted on the sessions TR windows can see (no leakage) — inverse_transform in segment 10
        last_tr_row = int(ends[parts['TR'][-1]])
        mu, sd = x_all[:last_tr_row + 1].mean(0), x_all[:last_tr_row + 1].std(0) + 1e-8
        y_mu, y_sd = float(sy[parts['TR']].mean()), float(sy[parts['TR']].std()) or 1.0
        SX = ((sx - mu.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32)
        SY = ((sy - y_mu) / y_sd).astype(np.float32)
        inverse = lambda v: np.asarray(v) * y_sd + y_mu  # noqa: E731  (y_scaler.inverse_transform)
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        rows: list[dict[str, Any]] = []
        histories: dict[str, list[dict[str, float]]] = {}
        for i, name in enumerate(p.candidates):
            seed_all(context.seed)
            model = RecurrentRegressor('GRU' if name.startswith('GRU') else 'LSTM', SX.shape[2], p.hidden_size,
                                       name.endswith('(bi)'), p.rnn_layers, p.stack()).to(self.device)
            tr = np.concatenate([parts['TR'], parts['VA']])          # fit() keeps the tail (= VA) for early stopping
            training = fit(model, t(SX[tr]), t(SY[tr]), nn.functional.mse_loss, p, seed=context.seed,
                           val_fraction=len(parts['VA']) / len(tr))
            with torch.no_grad():
                va = inverse(model(t(SX[parts['VA']])).cpu().numpy())
            mse = float(np.mean((va - sy[parts['VA']]) ** 2))
            rows.append({'modelo': name, 'validation_MSE': mse, 'parámetros': sum(q.numel() for q in model.parameters()),
                         'épocas': training['epochs_run'], 'mejor_época': training['best_epoch']})
            histories[name] = training['history']
            self.model[name.replace(' ', '_').replace('(', '').replace(')', '')] = model
        # segment 09: selected = comparison.sort_values('validation_MSE').iloc[0].modelo
        rows.sort(key=lambda r: r['validation_MSE'])
        selected = rows[0]['modelo']
        best = self.model[selected.replace(' ', '_').replace('(', '').replace(')', '')]
        with torch.no_grad():
            pred = inverse(best(t(SX[parts['TE']])).cpu().numpy())
            live = float(inverse(best(t(SX[-1:])).cpu().numpy())[0]) if not known[-1] else None
        # Ridge on flattened windows, same scaled inputs, fit on TR only
        flat = SX.reshape(len(SX), -1)
        ridge = Ridge(alpha=p.ridge_alpha).fit(flat[parts['TR']], SY[parts['TR']])
        ridge_va = inverse(ridge.predict(flat[parts['VA']]))
        ridge_te = inverse(ridge.predict(flat[parts['TE']]))
        actual = sy[parts['TE']]
        zero = float(np.mean(actual ** 2))
        te = {'selected_MSE': float(np.mean((pred - actual) ** 2)), 'ridge_MSE': float(np.mean((ridge_te - actual) ** 2)),
              'zero_MSE': zero, 'mean_MSE': float(np.mean((actual - sy[parts['TR']].mean()) ** 2))}
        hit = lambda a: float(np.mean(np.sign(a) == np.sign(actual)))  # noqa: E731
        rows.append({'modelo': 'Ridge (línea base)', 'validation_MSE': float(np.mean((ridge_va - sy[parts['VA']]) ** 2)),
                     'parámetros': int(flat.shape[1] + 1), 'épocas': None, 'mejor_época': None})
        outputs = {'prediction': pred.tolist(), 'actual': actual.tolist(), 'next': live, 'comparison': rows, 'selected': selected}
        diag = {'shape': [int(len(SX)), p.lookback, int(SX.shape[2])], 'splits': {k: len(v) for k, v in parts.items()},
                'test': te, 'hit_selected': hit(pred), 'hit_ridge': hit(ridge_te), 'histories': histories,
                'te_rows': [int(ends[i]) for i in parts['TE']], 'ridge_te': ridge_te.tolist(), 'live_row': int(ends[-1]) if live is not None else None}
        self.weights = 'fitted_on_input'
        return outputs, diag
