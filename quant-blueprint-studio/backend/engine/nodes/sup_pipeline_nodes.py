"""Unidades 2–7 · Aprendizaje supervisado: herramientas atómicas derivadas de las infografías.

Sustituyen al nodo monolítico ``study`` (que entrenaba, seleccionaba y evaluaba todo dentro de una
función según ``topic``). Cada herramienta hace UNA cosa, recibe y devuelve el pin ``frame`` y deja su
resultado en el estado compartido (:mod:`pipeline_state`). El Blueprint decide qué herramientas usar y
en qué orden; ninguna herramienta importa a otra.

Catálogo (segmento de infografía que cubre):

=====================  ==========================================================================
sup_target             U2-03 objetivo intradía t+1 en bps · U4-01 etiqueta de dirección
sup_split              U2-04 · U3-01 · U5-01 · U6-03 · U10-01 separación cronológica TR/VA/TE
sup_scale              U2-05 · U3-01 · U6-01 · U10-01 escaladores ajustados sólo con TR
sup_windows            U6-02/04 · U7-01 ventanas (batch, tiempo, rasgos)
sup_init               U3-02 · U4-02 pesos iniciales comunes (Kaiming)
sup_mlp                U2-06 · U4-02 · U6-08 MLP / referencia aplanada
sup_cnn                U6-05/06/07 Conv1d → Conv1d → pooling global → lineal
sup_rnn                U7-02…07 LSTM / GRU / bidireccional → estado final → lineal
sup_optimizer          U3-03 variante con SGD / RMSprop / Adam / AdamW
sup_regularizer        U4-03…07 variante con Dropout / L1 / L2 / BatchNorm / Early stopping
sup_loss               U2-07 · U4-08 MSE / Huber / BCE con logits
sup_hpo                U5-02…08 búsqueda aleatoria con folds causales y escaladores por fold
sup_train              U2-08/09 · U3-04…08 · U7-08 entrenamiento con checkpoint en VA
sup_select             U3-09 · U4-09 · U6-09 · U7-09 selección por validación
sup_evaluate           U2-10 · U3-10 · U4-10 · U6-10 · U7-10 test, Ridge y resultado con costes
=====================  ==========================================================================

Reglas sin fuga: rasgos al cierre t, objetivo en la sesión t+1; escaladores, pesos y Ridge sólo con
TR; VA elige checkpoint y modelo; TE se mide una sola vez en ``sup_evaluate``.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import pipeline_state as S
from .scaling_nodes import affine_audit, fit_scaler

FAMILY = 'sup'
DEFAULT_FEATURES = 'LogReturn,Volatility,RSI,EMA,SMA,VWAP,MACD,MACDSignal,BBUpper,BBMiddle,BBLower'
ACTS: dict[str, type[nn.Module]] = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'GELU': nn.GELU, 'SiLU': nn.SiLU, 'LeakyReLU': nn.LeakyReLU}
LABEL = {
    'sup_target': 'Definir el objetivo', 'sup_split': 'Separar el tiempo', 'sup_scale': 'Escalar sólo con train',
    'sup_windows': 'Construir ventanas', 'sup_init': 'Fijar la inicialización', 'sup_mlp': 'Construir la MLP',
    'sup_cnn': 'Construir la CNN 1D', 'sup_rnn': 'Construir la red recurrente', 'sup_optimizer': 'Definir el optimizador',
    'sup_regularizer': 'Añadir regularización', 'sup_loss': 'Elegir la pérdida', 'sup_hpo': 'Búsqueda de hiperparámetros',
    'sup_train': 'Entrenar con checkpoint', 'sup_select': 'Seleccionar con validación', 'sup_evaluate': 'Evaluar en test',
}


def _state(frame: pd.DataFrame, need: str | None = None) -> dict[str, Any]:
    return S.state(frame, FAMILY, need, LABEL)


def _emit(frame: pd.DataFrame, st: dict[str, Any], step: str, out: pd.DataFrame | None = None) -> pd.DataFrame:
    return S.emit(frame, FAMILY, st, step, out)


# ── 1 · Definir el objetivo ───────────────────────────────────────────────────────────────────────
def target(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    """Rasgos conocidos al cierre t; objetivo = intradía de la sesión t+1 (bps) o su dirección (0/1)."""
    features = S.names(p.get('features')) or S.names(DEFAULT_FEATURES)
    task = str(p.get('task', 'regression'))
    if task not in ('regression', 'direction'):
        raise ValueError('task debe ser regression o direction.')
    if len(set(features)) != len(features):
        raise ValueError('Hay indicadores repetidos en la lista.')
    lost = [c for c in [*features, 'Open', 'Close'] if c not in frame.columns]
    if lost:
        raise ValueError(f'Columnas inexistentes: {", ".join(lost)}. Activa sus indicadores en Technical Features.')
    nxt_open, nxt_close = frame['Open'].shift(-1).astype(float), frame['Close'].shift(-1).astype(float)
    data = frame[features].apply(pd.to_numeric, errors='coerce').astype(float)
    data['y_bps'] = 1e4 * np.log(nxt_close / nxt_open)
    data['gap'] = np.log(nxt_open / frame['Close'].astype(float))
    data['target_date'] = pd.Series(frame.index, index=frame.index).shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 60:
        raise ValueError(f'Sólo {len(data)} filas útiles: amplía la historia.')
    y = data['y_bps'].to_numpy(float)
    st = {'X_raw': data[features].to_numpy(np.float64), 'names': features, 'y_bps': y, 'direction': (y > 0).astype(np.float32),
          'intraday': y / 1e4, 'gap': data['gap'].to_numpy(float), 'dates': data.index, 'target_dates': pd.DatetimeIndex(data['target_date']),
          'task': task}
    out = frame.copy()
    out['y_bps'] = data['y_bps'].reindex(out.index)
    if task == 'direction':
        out['Direction'] = (out['y_bps'] > 0).astype(float).where(out['y_bps'].notna())
    rows = [{'date': S.day(data.index[i]), 'y_bps': float(y[i])} for i in S.thin(len(y))]
    return _emit(frame, st, 'sup_target', out), S.report(
        'sup_target', selected=f'{len(features)} rasgos · {"dirección (y_bps > 0)" if task == "direction" else "y_bps intradía t+1"}', features=features,
        metrics={'filas': len(y), 'media_y_bps': float(y.mean()), 'σ_y_bps': float(y.std()), 'fracción_alza': float((y > 0).mean())},
        charts=[S.chart('Objetivo · 10000·log(Close[t+1] / Open[t+1])', rows, ['y_bps'], 'date')],
        notes=["f['y_bps'] = 10000 * np.log(df.Close.shift(-1) / df.Open.shift(-1)): señal al cierre t, entrada en la apertura t+1 y salida al cierre t+1.",
               *(["panel['direction'] = (panel.y_bps > 0): la clase 1 es una sesión alcista."] if task == 'direction' else []),
               'Se conserva también el gap nocturno log(Open[t+1]/Close[t]) para el entorno de trading (Unidad 10).'])


# ── 2 · Separar el tiempo ─────────────────────────────────────────────────────────────────────────
def split(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_target')
    dates, tdates = st['dates'], st['target_dates']
    if str(p.get('cut_mode', 'ratio')) == 'dates':
        vs, ts = pd.Timestamp(str(p.get('val_start'))), pd.Timestamp(str(p.get('test_start')))
    else:
        tr, va = float(p.get('train_ratio', .65)), float(p.get('val_ratio', .15))
        if not .3 <= tr <= .9 or not .05 <= va <= .4 or tr + va >= .95:
            raise ValueError('TR 0.3–0.9, VA 0.05–0.4 y TR + VA < 0.95.')
        vs, ts = dates[int(tr * len(dates))], dates[int((tr + va) * len(dates))]
    if not vs < ts:
        raise ValueError('VAL_START debe ser anterior a TEST_START.')
    train = (dates < vs) & (tdates < vs)
    valid = (dates >= vs) & (tdates < ts)
    test = dates >= ts
    g = np.where(train, 0, np.where(valid, 1, np.where(test, 2, -1)))
    counts = {k: int((g == i).sum()) for i, k in enumerate(('train', 'validation', 'test'))}
    if min(counts.values()) < 8:
        raise ValueError(f'Cada bloque necesita ≥ 8 filas ({counts}). Amplía la historia o mueve los cortes.')
    st = {**st, 'group': g, 'cuts': {'VAL_START': S.day(vs), 'TEST_START': S.day(ts)}}
    out = frame.copy()
    out['Split'] = np.where(out.index < vs, 'TR', np.where(out.index < ts, 'VA', 'TE'))
    return _emit(frame, st, 'sup_split', out), S.report(
        'sup_split', selected=f'VAL_START {S.day(vs)} · TEST_START {S.day(ts)}', splits=counts,
        metrics={**counts, 'purgadas': int((g < 0).sum())},
        table=[{'bloque': k, 'filas': v} for k, v in counts.items()] + [{'bloque': 'purgadas (objetivo cruza el corte)', 'filas': int((g < 0).sum())}],
        notes=["tr = panel[(panel.signal_date < VAL_START) & (panel.target_date < VAL_START)]: ninguna etiqueta de TR pertenece a VA/TE.",
               'Orden cronológico sin barajar: train → validación → test.'])


# ── 3 · Escalar sólo con train ────────────────────────────────────────────────────────────────────
def scale(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_split')
    method = str(p.get('method', 'standard'))
    tr = st['group'] == 0
    xs, x_audit = fit_scaler(st['X_raw'][tr], st['names'], method=method, title='x_scaler · entradas')
    X = xs.transform(st['X_raw']).astype(np.float32)
    y = st['y_bps']
    scale_y = bool(p.get('scale_target', True)) and st['task'] == 'regression'
    y_mean, y_std = (float(y[tr].mean()), max(float(y[tr].std()), 1e-8)) if scale_y else (0., 1.)
    audits = [x_audit]
    if scale_y:
        audits.append(affine_audit(y[tr], y_mean, y_std, 'y_scaler · objetivo y_bps', scope='training'))
    st = {**st, 'X': X, 'x_scaler': xs, 'y_mean': y_mean, 'y_std': y_std, 'scale_method': method}
    va = st['group'] == 1
    table = [{'rasgo': n, 'media TR': float(xs.mean_[i]) if hasattr(xs, 'mean_') else float(xs.data_min_[i]),
              'media VA escalada': float(X[va, i].mean())} for i, n in enumerate(st['names'])]
    return _emit(frame, st, 'sup_scale'), S.report(
        'sup_scale', selected=f'{type(xs).__name__} · fit TR' + (' · y_scaler' if scale_y else ''), features=st['names'], splits=S.splits(st),
        table=table, scaling_audits=audits, metrics={'filas_ajuste': int(tr.sum()), 'y_mean': y_mean, 'y_std': y_std},
        notes=["x_scaler.fit(tr[FEATURES]) · y_scaler.fit(tr[['y_bps']]): VA y TE sólo se transforman.",
               'La media escalada de VA no es exactamente 0: es la señal de que no hubo fuga.'])


# ── 4 · Construir ventanas ────────────────────────────────────────────────────────────────────────
def windows(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_scale')
    L = int(p.get('lookback', 20))
    if not 1 <= L <= 250:
        raise ValueError('LOOKBACK debe estar entre 1 y 250 sesiones.')
    X = st['X']
    rows = np.arange(L - 1, len(X))
    X3 = np.stack([X[j - L + 1:j + 1] for j in rows]).astype(np.float32)
    st = {**st, 'X3': X3, 'rows': rows, 'lookback': L}
    g = st['group'][rows]
    counts = {'train': int((g == 0).sum()), 'validation': int((g == 1).sum()), 'test': int((g == 2).sum())}
    if min(counts.values()) < 8:
        raise ValueError(f'Con LOOKBACK {L} quedan {counts}: reduce la ventana o amplía la historia.')
    return _emit(frame, st, 'sup_windows'), S.report(
        'sup_windows', selected=f'(batch, {L}, {X.shape[1]})', features=st['names'], splits=counts,
        metrics={'ventanas': len(X3), 'lookback': L, 'rasgos': X.shape[1], 'descartadas_sin_historia': L - 1},
        notes=['window = values[j-LOOKBACK+1:j+1]: la ventana termina en el cierre j y su etiqueta es la sesión j+1.',
               'Cada ventana se asigna a TR/VA/TE por la fecha de su etiqueta (la última fila).'])


def _design(st: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tensor (n, L, f) y filas base a las que corresponde cada muestra."""
    if 'X3' in st:
        return st['X3'], st['rows']
    return st['X'][:, None, :], np.arange(len(st['X']))


# ── 5 · Fijar la inicialización ───────────────────────────────────────────────────────────────────
def init(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_split')
    scheme = str(p.get('scheme', 'kaiming_normal'))
    if scheme not in ('kaiming_normal', 'kaiming_uniform', 'xavier_uniform', 'pytorch_default'):
        raise ValueError('Esquema de inicialización desconocido.')
    st = {**st, 'init': {'scheme': scheme, 'seed': int(p.get('seed', 42))}}
    return _emit(frame, st, 'sup_init'), S.report(
        'sup_init', selected=f'{scheme} · semilla {int(p.get("seed", 42))}', splits=S.splits(st),
        notes=["nn.init.kaiming_normal_(layer.weight, nonlinearity='relu') · initial_state = copy.deepcopy(model.state_dict()).",
               'Todas las variantes de una misma arquitectura arrancan de los mismos pesos: las diferencias se deben al optimizador o a la regularización.'])


# ── Arquitecturas ─────────────────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """Red construida desde una especificación declarativa (nunca código del grafo)."""

    def __init__(self, spec: dict[str, Any], n_features: int, lookback: int) -> None:
        super().__init__()
        self.kind = spec['kind']
        act = ACTS[spec.get('activation', 'ReLU')]
        if self.kind == 'CNN':
            (c1, c2), k = spec['channels'], int(spec.get('kernel', 3))
            self.encoder = nn.Sequential(nn.Conv1d(n_features, c1, k, padding=k // 2), act(), nn.Conv1d(c1, c2, k, padding=k // 2), act(),
                                         nn.AdaptiveAvgPool1d(1), nn.Flatten())
            d = c2
        elif self.kind == 'RNN':
            cell = nn.LSTM if spec['cell'] == 'LSTM' else nn.GRU
            self.bidirectional = bool(spec.get('bidirectional'))
            self.encoder = cell(n_features, int(spec['hidden']), batch_first=True, bidirectional=self.bidirectional)
            d = int(spec['hidden']) * (2 if self.bidirectional else 1)
        else:
            self.encoder = nn.Flatten()
            d = n_features * lookback
        layers: list[nn.Module] = []
        for w in spec.get('widths', []) if self.kind == 'MLP' else []:
            layers.append(nn.Linear(d, w))
            if spec.get('batchnorm'):
                layers.append(nn.BatchNorm1d(w))
            layers.append(act())
            if spec.get('dropout'):
                layers.append(nn.Dropout(float(spec['dropout'])))
            d = w
        self.head = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == 'CNN':
            h = self.encoder(x.transpose(1, 2))  # (batch, tiempo, rasgos) → (batch, canales, tiempo)
        elif self.kind == 'RNN':
            _, state = self.encoder(x)  # cada ventana empieza con memoria nueva
            h = state[0] if isinstance(state, tuple) else state  # LSTM → (h, c); GRU → h
            h = torch.cat([h[-2], h[-1]], 1) if self.bidirectional else h[-1]
        else:
            h = self.encoder(x)
        return self.head(h).squeeze(1)


def _arch(spec: dict[str, Any], st: dict[str, Any]) -> list[str]:
    X, _ = _design(st)
    net = Net(spec, X.shape[-1], X.shape[1])
    return [f'{name} · {type(m).__name__}' for name, m in net.named_modules() if name and not list(m.children())]


def _add(frame: pd.DataFrame, p: dict[str, Any], st: dict[str, Any], spec: dict[str, Any], step: str, formula: list[str]):
    include = bool(p.get('include_as_candidate', True))
    cands = [*st.get('candidates', []), spec] if include else list(st.get('candidates', []))
    base = st.get('base') or spec
    st = {**st, 'candidates': cands, 'base': base}
    X, _ = _design(st)
    params = sum(v.numel() for v in Net(spec, X.shape[-1], X.shape[1]).parameters())
    return _emit(frame, st, step), S.report(
        step, selected=spec['name'], splits=S.splits(st), metrics={'parámetros': params, 'candidatos': len(cands)},
        table=[{'capa': layer} for layer in _arch(spec, st)],
        notes=[*formula, 'Candidato añadido a la comparación.' if include else 'Plantilla base: las variantes (optimizador / regularización) parten de ella; no compite por sí sola.'])


def mlp(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_scale')
    widths = S.ints(p.get('widths', '64,32'), 1, 1024, 'Anchos ocultos')
    if len(widths) > 6:
        raise ValueError('Como máximo 6 capas ocultas.')
    act = str(p.get('activation', 'ReLU'))
    spec = {'kind': 'MLP', 'widths': widths, 'activation': act, 'dropout': float(p.get('dropout', 0)), 'batchnorm': bool(p.get('batchnorm', False)),
            'name': str(p.get('name') or 'MLP ' + '→'.join(map(str, widths)))}
    X, _ = _design(st)
    return _add(frame, p, st, spec, 'sup_mlp', [
        f'nn.Linear({X.shape[-1] * X.shape[1]}, {widths[0]}), nn.{act}(), …, nn.Linear({widths[-1]}, 1)',
        'Con ventanas, la MLP las aplana: la misma información que la CNN/RNN, sin filtros compartidos.' if X.shape[1] > 1 else 'Una fila de rasgos por muestra.'])


def cnn(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_windows')
    ch = S.ints(p.get('channels', '32,64'), 1, 512, 'Canales')
    if len(ch) != 2:
        raise ValueError('Canales: dos valores (primer y segundo filtro).')
    k = int(p.get('kernel', 3))
    if not 1 <= k <= st['lookback']:
        raise ValueError('El kernel no puede superar LOOKBACK.')
    spec = {'kind': 'CNN', 'channels': ch, 'kernel': k, 'activation': str(p.get('activation', 'ReLU')), 'name': str(p.get('name') or f'CNN_1D {ch[0]}→{ch[1]}')}
    return _add(frame, p, st, spec, 'sup_cnn', [
        f'nn.Conv1d(len(FEATURES), {ch[0]}, {k}) → nn.Conv1d({ch[0]}, {ch[1]}, {k}) → nn.AdaptiveAvgPool1d(1) → nn.Flatten() → nn.Linear({ch[1]}, 1)',
        'x.transpose(1, 2): Conv1d recibe (batch, canales = rasgos, tiempo).'])


def rnn(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_windows')
    cell, hidden, bi = str(p.get('cell', 'LSTM')), int(p.get('hidden', 32)), bool(p.get('bidirectional', False))
    if cell not in ('LSTM', 'GRU') or not 2 <= hidden <= 512:
        raise ValueError('Celda LSTM o GRU y 2–512 unidades.')
    spec = {'kind': 'RNN', 'cell': cell, 'hidden': hidden, 'bidirectional': bi, 'name': str(p.get('name') or f'{cell}{" (bi)" if bi else ""}')}
    return _add(frame, p, st, spec, 'sup_rnn', [
        f'nn.{cell}(len(FEATURES), {hidden}, batch_first=True{", bidirectional=True" if bi else ""})',
        ("h = state[0] (LSTM devuelve (h, c))" if cell == 'LSTM' else 'h = state (GRU devuelve sólo h)') + (' · summary = torch.cat([h[-2], h[-1]], dim=1)' if bi else ' · summary = h[-1]'),
        f'self.head = nn.Linear({hidden * (2 if bi else 1)}, 1)'])


# ── Variantes de entrenamiento ────────────────────────────────────────────────────────────────────
def _variant(frame: pd.DataFrame, st: dict[str, Any], spec: dict[str, Any], step: str, notes: list[str]):
    st = {**st, 'candidates': [*st.get('candidates', []), spec]}
    return _emit(frame, st, step), S.report(step, selected=spec['name'], splits=S.splits(st),
                                            metrics={'candidatos': len(st['candidates'])}, table=[{k: str(v) for k, v in spec.items()}], notes=notes)


def optimizer(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_mlp')
    name, lr = str(p.get('optimizer', 'AdamW')), float(p.get('lr', .001))
    if name not in ('SGD', 'RMSprop', 'Adam', 'AdamW') or not 1e-6 <= lr <= 1:
        raise ValueError('Optimizador SGD/RMSprop/Adam/AdamW y lr 1e-6–1.')
    spec = {**st['base'], 'optimizer': name, 'lr': lr, 'momentum': float(p.get('momentum', 0)), 'name': f'{name} (lr {lr:g})'}
    return _variant(frame, st, spec, 'sup_optimizer', [
        f"configs = {{'{name}': {lr:g}}} · misma arquitectura y mismos pesos iniciales que la plantilla.",
        'optimizer.zero_grad(set_to_none=True) vacía los gradientes del batch anterior antes de cada paso.'])


def regularizer(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_mlp')
    kind, value = str(p.get('kind', 'Dropout')), float(p.get('value', .3))
    base = st['base']
    change: dict[str, Any] = {
        'Baseline': {}, 'Dropout': {'dropout': value}, 'L1': {'l1': value}, 'L2': {'l2': value},
        'BatchNorm': {'batchnorm': True}, 'EarlyStopping': {'early': int(value)},
    }.get(kind)
    if change is None:
        raise ValueError('Regularización: Baseline, Dropout, L1, L2, BatchNorm o EarlyStopping.')
    if kind == 'Dropout' and not 0 < value < 1 or kind in ('L1', 'L2') and not 0 < value <= .1 or kind == 'EarlyStopping' and not 1 <= value <= 200:
        raise ValueError('Dropout 0–1, L1/L2 0–0.1, paciencia 1–200.')
    label = kind if kind in ('Baseline', 'BatchNorm') else f'{kind} {value:g}'
    formula = {'Baseline': 'Misma MLP sin regularización: referencia común.', 'Dropout': f'nn.Dropout({value:g}) tras cada activación oculta (sólo en train).',
               'L1': f'loss = data_loss + {value:g} * sum(|w|) de los pesos (no sesgos).', 'L2': f'torch.optim.Adam(..., weight_decay={value:g}).',
               'BatchNorm': 'nn.BatchNorm1d(width) tras cada capa lineal oculta; model.train()/eval() cambia sus estadísticas.',
               'EarlyStopping': f'patience = {int(value)} · restore_best = True: se restauran los pesos con mejor pérdida de validación.'}[kind]
    return _variant(frame, st, {**base, **change, 'name': label}, 'sup_regularizer', [formula, 'La gráfica de pérdidas compara la pérdida de datos, sin penalizaciones.'])


def loss(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_target')
    name = str(p.get('loss', 'auto'))
    if name == 'auto':
        name = 'BCE' if st['task'] == 'direction' else 'MSE'
    if name not in ('MSE', 'Huber', 'BCE'):
        raise ValueError('Pérdida MSE, Huber o BCE.')
    if (name == 'BCE') != (st['task'] == 'direction'):
        raise ValueError(f'La pérdida {name} no corresponde al objetivo «{st["task"]}»: usa BCE para dirección y MSE/Huber para magnitud.')
    st = {**st, 'loss': name}
    return _emit(frame, st, 'sup_loss'), S.report('sup_loss', selected=name, splits=S.splits(st), notes=[
        "loss_fn = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()",
        {'MSE': 'MSE = (1/n)·Σ(ŷ − y)² mide magnitud.', 'Huber': 'Huber: cuadrática cerca de 0, lineal en colas; menos sensible a días extremos.',
         'BCE': 'BCE con logits: −(1/n)·Σ[y·log σ(z) + (1−y)·log(1−σ(z))], numéricamente estable.'}[name]])


# ── Entrenamiento compartido ──────────────────────────────────────────────────────────────────────
def _build(spec: dict[str, Any], st: dict[str, Any], X: np.ndarray) -> Net:
    ini = st.get('init') or {'scheme': 'kaiming_uniform', 'seed': 42}
    torch.manual_seed(int(ini['seed']))  # mismo arranque para variantes de la misma arquitectura
    net = Net(spec, X.shape[-1], X.shape[1])
    for m in net.modules():
        if isinstance(m, nn.Linear) and ini['scheme'] != 'pytorch_default':
            {'kaiming_normal': lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
             'kaiming_uniform': lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
             'xavier_uniform': nn.init.xavier_uniform_}[ini['scheme']](m.weight)
            nn.init.zeros_(m.bias)
    return net


def _labels(st: dict[str, Any], rows: np.ndarray, mean: float, std: float) -> np.ndarray:
    if st['task'] == 'direction':
        return st['direction'][rows].astype(np.float32)
    return ((st['y_bps'][rows] - mean) / std).astype(np.float32)


def _loss_fn(name: str):
    return {'MSE': F.mse_loss, 'Huber': F.huber_loss, 'BCE': F.binary_cross_entropy_with_logits}[name]


def _fit(spec: dict[str, Any], st: dict[str, Any], X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, p: dict[str, Any],
         progress: S.Progress, dev: torch.device) -> tuple[Net, list[dict[str, Any]], float, int]:
    net = _build(spec, st, X).to(dev)
    lr = float(spec.get('lr', p.get('lr', .001)))
    opt_cls = {'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop, 'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[spec.get('optimizer', 'Adam')]
    kw: dict[str, Any] = {'lr': lr, 'weight_decay': float(spec.get('l2', 0))}
    if spec.get('optimizer') == 'SGD' and spec.get('momentum'):
        kw['momentum'] = float(spec['momentum'])
    opt = opt_cls(net.parameters(), **kw)
    fn = _loss_fn(st.get('loss', 'BCE' if st['task'] == 'direction' else 'MSE'))
    xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
    xtr, ytr, xva, yva = xt[tr], yt[tr], xt[va], yt[va]
    epochs, bs, clip = int(p.get('epochs', 30)), int(p.get('batch_size', 64)), float(p.get('grad_clip', 1.))
    early = spec.get('early', int(p.get('patience', 0)) or None)
    restore = bool(spec.get('early')) or bool(p.get('restore_best', True))
    amp = bool(p.get('amp', False)) and dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    gen = torch.Generator().manual_seed(int(p.get('seed', 42)))
    best, best_state, best_epoch, stale, hist = np.inf, deepcopy(net.state_dict()), 0, 0, []
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(xtr), generator=gen).split(bs):
            if len(idx) < 2 and spec.get('batchnorm'):
                continue
            idx = idx.to(dev)
            opt.zero_grad(set_to_none=True)                       # vaciar gradientes anteriores
            with torch.autocast(dev.type, enabled=amp):
                pred = net(xtr[idx])                              # salida
            data_loss = fn(pred.float(), ytr[idx].float())        # pérdida de datos
            penalty = sum(w.abs().sum() for n, w in net.named_parameters() if n.endswith('weight')) if spec.get('l1') else 0.
            total = data_loss + float(spec.get('l1', 0)) * penalty
            S.finite('pérdida', total)
            scaler.scale(total).backward()                        # retropropagar
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), clip)      # limitar norma
            scaler.step(opt)                                      # actualizar pesos
            scaler.update()
        net.eval()
        with torch.no_grad():
            tr_loss, va_loss = float(fn(net(xtr).float(), ytr)), float(fn(net(xva).float(), yva))
        S.finite('pérdida de validación', va_loss)
        hist.append({'epoch': ep + 1, 'train': tr_loss, 'validation': va_loss})
        if progress:
            progress({'epoch': ep + 1, 'epochs': epochs, 'train': tr_loss, 'validation': va_loss, 'detail': spec['name']})
        if va_loss < best:
            best, best_state, best_epoch, stale = va_loss, deepcopy(net.state_dict()), ep + 1, 0
        else:
            stale += 1
        if early and stale >= early:
            break
    if restore:
        net.load_state_dict(best_state)
    else:
        best, best_epoch = hist[-1]['validation'], len(hist)
    return net.cpu().eval(), hist, best, best_epoch


def _masks(st: dict[str, Any], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = st['group'][rows]
    return np.flatnonzero(g == 0), np.flatnonzero(g == 1), np.flatnonzero(g == 2)


# ── HPO · búsqueda con folds causales ─────────────────────────────────────────────────────────────
def hpo(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_split')
    if st['task'] != 'regression':
        raise ValueError('La búsqueda HPO de la Unidad 5 mide RMSE en bps: usa un objetivo de regresión.')
    rng = np.random.default_rng(int(p.get('seed', 42)))
    widths, depths = S.ints(p.get('width_choices', '32,64,128'), 2, 1024, 'Anchos'), S.ints(p.get('depth_choices', '1,2,3'), 1, 6, 'Profundidades')
    lo, hi = float(p.get('log_lr_min', -4)), float(p.get('log_lr_max', -2.5))
    wds = [float(v) for v in S.names(p.get('wd_choices', '0,0.0001,0.001'))]
    fracs = [float(v) for v in S.names(p.get('folds', '0.5,0.75'))]
    trials = int(p.get('trials', 6))
    if not lo < hi or not 1 <= trials <= 50 or not fracs or any(not .2 <= f <= .9 for f in fracs):
        raise ValueError('log10(lr) mín < máx, 1–50 pruebas y folds entre 0.2 y 0.9 de TR.')
    L = int(st.get('lookback', 1))
    Xr = st['X_raw']
    rows = np.arange(L - 1, len(Xr))
    tr_rows = rows[st['group'][rows] == 0]
    fit_p = {**p, 'patience': int(p.get('patience', 5)), 'restore_best': True}
    table, t0 = [], time.perf_counter()
    for t in range(trials):
        cfg = {'kind': 'MLP', 'widths': [int(rng.choice(widths))] * int(rng.choice(depths)), 'lr': float(10 ** rng.uniform(lo, hi)),
               'l2': float(rng.choice(wds)), 'activation': 'ReLU', 'name': f'Trial {t + 1}'}
        row: dict[str, Any] = {'trial': t + 1, 'width': cfg['widths'][0], 'depth': len(cfg['widths']), 'lr': cfg['lr'], 'weight_decay': cfg['l2']}
        scores = []
        for k, f in enumerate(fracs):
            a = int(len(tr_rows) * f)
            b = min(len(tr_rows), a + max(int(len(tr_rows) * .25), 8))
            fit_rows, val_rows = tr_rows[:a], tr_rows[a:b]
            if len(fit_rows) < 16 or len(val_rows) < 8:
                raise ValueError('Folds demasiado pequeños: amplía la historia o cambia las fracciones.')
            xs = StandardScaler().fit(Xr[fit_rows])                   # reajuste por fold: sólo su pasado
            ym, ysd = float(st['y_bps'][fit_rows].mean()), max(float(st['y_bps'][fit_rows].std()), 1e-8)
            Xs = xs.transform(Xr).astype(np.float32)
            X3 = np.stack([Xs[j - L + 1:j + 1] for j in np.r_[fit_rows, val_rows]])
            y = ((st['y_bps'][np.r_[fit_rows, val_rows]] - ym) / ysd).astype(np.float32)
            net, _, _, _ = _fit(cfg, {**st, 'loss': 'MSE'}, X3, y, np.arange(len(fit_rows)), np.arange(len(fit_rows), len(y)), fit_p, None, torch.device('cpu'))
            with torch.no_grad():
                pred = net(torch.tensor(X3[len(fit_rows):])).numpy() * ysd + ym      # medir en bps reales
            rmse = float(np.sqrt(mean_squared_error(st['y_bps'][val_rows], pred)))
            row[f'fold {k + 1} RMSE (bps)'] = rmse
            scores.append(rmse)
            if progress:
                progress({'detail': f'Trial {t + 1}/{trials} · fold {k + 1}', 'trial': t + 1})
        row['RMSE medio (bps)'] = float(np.mean(scores))
        cfg['cv_rmse'] = row['RMSE medio (bps)']
        table.append((row, cfg))
    best_row, best_cfg = min(table, key=lambda rc: rc[0]['RMSE medio (bps)'])
    spec = {**best_cfg, 'name': 'HPO ' + '→'.join(map(str, best_cfg['widths']))}
    st = {**st, 'candidates': [*st.get('candidates', []), spec], 'base': st.get('base') or spec, 'hpo_trials': [r for r, _ in table]}
    return _emit(frame, st, 'sup_hpo'), S.report(
        'sup_hpo', selected=f'{spec["name"]} · lr {spec["lr"]:.2e} · wd {spec["l2"]:g}', splits=S.splits(st), table=[r for r, _ in table],
        metrics={'mejor RMSE medio (bps)': best_row['RMSE medio (bps)'], 'pruebas': trials, 'folds': len(fracs), 'segundos': time.perf_counter() - t0},
        charts=[S.chart('RMSE medio por prueba (bps)', [{'trial': r['trial'], 'RMSE medio (bps)': r['RMSE medio (bps)']} for r, _ in table], ['RMSE medio (bps)'], 'trial', 'bar')],
        notes=['width = rng.choice(widths) · depth = rng.choice(depths) · lr = 10 ** rng.uniform(lo, hi) · weight_decay = rng.choice(wds).',
               'Folds expansivos dentro de TR (entrenar antes del bloque que se valida). x/y StandardScaler se reajustan en cada fold con su propio pasado.',
               'VA y TE no participan en la búsqueda; «Entrenar» ajusta después el modelo final con TR completo y checkpoint en VA.'])


# ── Entrenar con checkpoint ───────────────────────────────────────────────────────────────────────
def train(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_scale')
    cands = st.get('candidates') or []
    if not cands:
        raise ValueError('No hay candidatos: conecta una arquitectura (MLP / CNN / RNN), variantes o HPO antes de entrenar.')
    X, rows = _design(st)
    for c in cands:
        if c['kind'] in ('CNN', 'RNN') and 'X3' not in st:
            raise ValueError(f'{c["name"]} necesita «{LABEL["sup_windows"]}» aguas arriba.')
    tr, va, _ = _masks(st, rows)
    y = _labels(st, rows, st['y_mean'], st['y_std'])
    dev = S.device(p)
    results, table, curves, t0 = {}, [], {}, time.perf_counter()
    for c in cands:
        net, hist, best, ep = _fit(c, st, X, y, tr, va, p, progress, dev)
        unit = best * st['y_std'] ** 2 if st['task'] == 'regression' and st.get('loss', 'MSE') == 'MSE' else best
        results[c['name']] = {'net': net, 'history': hist, 'spec': c, 'validation': unit, 'epoch': ep}
        table.append({'modelo': c['name'], 'mejor época': ep, 'épocas': len(hist), 'parámetros': sum(v.numel() for v in net.parameters()),
                      ('validation_MSE (bps²)' if st['task'] == 'regression' else f'validation_{st.get("loss", "BCE")}'): unit})
        for h in hist:
            curves.setdefault(h['epoch'], {'epoch': h['epoch']})[c['name']] = h['validation']
    st = {**st, 'trained': results}
    charts = [S.chart('Pérdida de validación por época (datos, sin penalización)', sorted(curves.values(), key=lambda r: r['epoch']), [c['name'] for c in cands], 'epoch')]
    if len(cands) == 1:
        charts.insert(0, S.chart(f'{cands[0]["name"]} · pérdida train / validación', results[cands[0]['name']]['history'], ['train', 'validation'], 'epoch'))
    return _emit(frame, st, 'sup_train'), S.report(
        'sup_train', selected=f'{len(cands)} modelo(s) · {int(p.get("epochs", 30))} épocas · {S.device(p).type}', splits=S.splits(st),
        table=table, charts=charts, metrics={'segundos': time.perf_counter() - t0, 'candidatos': len(cands)},
        notes=['Por batch: zero_grad → pred = model(xb) → data_loss → scale(loss).backward() → unscale_ → clip_grad_norm_(…, 1.) → step → update.',
               'if row["validation"] < best: best_state = {k: v.detach().cpu().clone() …}: se guarda el checkpoint con menor pérdida en VA.',
               'Mismo presupuesto (épocas, batch, datos) para todos los candidatos. AMP sólo en CUDA; en CPU el escalador queda desactivado.'])


# ── Seleccionar con validación ────────────────────────────────────────────────────────────────────
def select(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_train')
    res = st['trained']
    ranking = sorted(res.items(), key=lambda kv: kv[1]['validation'])
    sel = ranking[0][0]
    st = {**st, 'selected': sel}
    col = 'validation_MSE (bps²)' if st['task'] == 'regression' else 'validation_BCE'
    table = [{'modelo': n, col: r['validation'], 'checkpoint (época)': r['epoch'], 'parámetros': sum(v.numel() for v in r['net'].parameters()),
              'seleccionado': n == sel} for n, r in ranking]
    return _emit(frame, st, 'sup_select'), S.report(
        'sup_select', selected=sel, splits=S.splits(st), table=table,
        charts=[S.chart('Comparación en validación', [{'modelo': n, col: r['validation']} for n, r in ranking], [col], 'modelo', 'bar')],
        notes=[f"selected = comparison.sort_values('{col}').iloc[0].modelo — test no se consulta."])


# ── Evaluar en test ───────────────────────────────────────────────────────────────────────────────
def evaluate(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_train')
    sel = st.get('selected') or min(st['trained'], key=lambda k: st['trained'][k]['validation'])
    net = st['trained'][sel]['net']
    X, rows = _design(st)
    tr, _, te = _masks(st, rows)
    direction = st['task'] == 'direction'
    with torch.no_grad():
        raw = net(torch.tensor(X[te])).numpy()
    if direction:
        pred = 1 / (1 + np.exp(-raw))                                   # sigmoid → probabilidad de alza
        actual = st['direction'][rows[te]].astype(float)
    else:
        pred = raw * st['y_std'] + st['y_mean']                          # y_scaler.inverse_transform
        actual = st['y_bps'][rows[te]]
    m: dict[str, float] = {}
    thr = float(p.get('threshold', .5 if direction else 0.))
    if direction:
        eps = 1e-7
        m.update({'Accuracy': float(np.mean((pred >= thr) == actual)), 'BCE': float(-np.mean(actual * np.log(pred + eps) + (1 - actual) * np.log(1 - pred + eps))),
                  'Brier': float(np.mean((pred - actual) ** 2)), 'Fracción alza real': float(actual.mean())})
    else:
        mse = float(mean_squared_error(actual, pred))
        m.update({'RMSE (bps)': float(np.sqrt(mse)), 'MAE (bps)': float(mean_absolute_error(actual, pred)), 'R²': float(r2_score(actual, pred)),
                  'Acierto de dirección': float(np.mean(np.sign(pred) == np.sign(actual)))})
        if p.get('ridge', True):
            ridge = Ridge(alpha=float(p.get('ridge_alpha', 1.))).fit(X[tr].reshape(len(tr), -1), st['y_bps'][rows[tr]])
            m['Ridge RMSE (bps)'] = float(np.sqrt(mean_squared_error(actual, ridge.predict(X[te].reshape(len(te), -1)))))
    # Resultado con costes: entrar en la apertura t+1 y salir al cierre t+1.
    short = bool(p.get('allow_short', True))
    signal = np.where(pred >= thr, 1, -1 if short else 0) if direction else np.where(pred > thr, 1, -1 if short else 0)
    cost = float(p.get('cost_bps', 5)) / 1e4
    net_ret = signal * np.expm1(st['intraday'][rows[te]]) - 2 * np.abs(signal) * cost
    equity, market = np.cumprod(1 + net_ret), np.cumprod(1 + np.expm1(st['intraday'][rows[te]]))
    sd = net_ret.std(ddof=1)
    m.update({'ROI neto': float(equity[-1] - 1), 'Sharpe': float(net_ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.,
              'Máx. drawdown': float(np.min(equity / np.maximum.accumulate(np.r_[1., equity])[1:] - 1)), 'Rotación': float(2 * np.abs(signal).sum())})
    dates = st['dates'][rows[te]]
    pr = [{'date': S.day(d), 'actual': float(a), 'predicted': float(b)} for d, a, b in zip(dates, actual, pred)]
    eq = [{'date': S.day(d), 'estrategia': float(e - 1), 'mercado intradía': float(k - 1)} for d, e, k in zip(dates, equity, market)]
    out = frame.copy()
    out['Prediction'] = pd.Series(pred, index=dates).reindex(out.index)
    return _emit(frame, {**st, 'selected': sel}, 'sup_evaluate', out), S.report(
        'sup_evaluate', selected=sel, scope='TE · medido una vez', splits=S.splits(st), metrics=m,
        table=[{'métrica': k, 'valor': v} for k, v in m.items()],
        charts=[S.chart('Probabilidad de alza en test' if direction else 'Intradía t+1 en test · bps', pr, ['actual', 'predicted'], 'date'),
                S.chart('Checkpoint seleccionado · pérdida', st['trained'][sel]['history'], ['train', 'validation'], 'epoch'),
                S.chart(f'Resultado en test · costes {p.get("cost_bps", 5)} bps por lado', eq, ['estrategia', 'mercado intradía'], 'date')],
        notes=['prob = torch.sigmoid(infer(models[selected], X["test"])) · señal = prob ≥ umbral.' if direction else
               "pred = y_scaler.inverse_transform(infer(models[selected], X['test']).reshape(-1, 1)).ravel()",
               'Ridge (sobre las mismas entradas aplanadas, fit TR) es una referencia lineal: no participa en la selección.' if not direction else
               'Un umbral fijo de 0.5 convierte la probabilidad en señal; Brier y BCE miden la calibración.',
               'Simulación educativa: entrada en apertura y salida al cierre de la sesión siguiente, con costes de ida y vuelta.'])


STEPS = {'sup_target': target, 'sup_split': split, 'sup_scale': scale, 'sup_windows': windows, 'sup_init': init, 'sup_mlp': mlp,
         'sup_cnn': cnn, 'sup_rnn': rnn, 'sup_optimizer': optimizer, 'sup_regularizer': regularizer, 'sup_loss': loss, 'sup_hpo': hpo,
         'sup_train': train, 'sup_select': select, 'sup_evaluate': evaluate}
