"""Submenú «17 / ML · REDES NEURONALES · TIME SERIES & YFINANCE»: herramientas desacopladas y conectables.

Cada herramienta hace UNA cosa, recibe y devuelve el pin ``frame`` y deja su resultado en el estado compartido
(:mod:`pipeline_state`, familia ``nn``). Ninguna herramienta importa a otra; el Blueprint decide el orden.

=================  ============================================================================================
nn_data            Ingesta yfinance con reintentos, caché en disco y fallback (caché / sintético)
nn_guard           Guardia transversal: NaN/Inf, fechas, duplicados, forma del estado y bitácora de errores
nn_input           Capa de entrada: 1 parámetro · polinomios · lags (k, stride) · activación · optimizador base
nn_split           TR / VA / TE cronológico con purga del horizonte · holdout o walk-forward
nn_scaler          Standard / MinMax / Robust / MaxAbs / pass-through · fit sólo TR · persistencia joblib
nn_embedding       Embeddings aprendidos: día de la semana, mes, régimen de volatilidad, ticker
nn_hidden          Capa oculta 32/64/128/256 duplicable: activación, optimizador, dropout, norm, L1/L2, residual
nn_output          Capa de salida: horizontes / clases, activación, pérdida, métricas, compilación
nn_train           Entrenamiento con checkpoint en VA, early stopping, guardia de gradientes y guardado
nn_finetune        Ajuste fino desde un modelo base (pin pretrained o archivo): freeze/unfreeze, lr reducido
nn_predict         Predicción TR/VA/TE y en vivo, desescalado, bandas conformales y exportación CSV
=================  ============================================================================================

Reglas sin fuga: rasgos conocidos al cierre t; objetivo en t+1..t+h; escaladores, umbrales de régimen y pesos
sólo con TR; VA elige checkpoint y calibra bandas; TE se mide una vez en ``nn_predict``.
"""
from __future__ import annotations

import io
import math
import os
import time
from copy import deepcopy
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from . import pipeline_state as S
from .yfinance_nodes import yfinance_data

FAMILY = 'nn'
ROOT = Path(os.environ.get('QL_ARTIFACTS', Path(__file__).resolve().parents[3] / 'artifacts' / 'nn'))
LABEL = {
    'nn_data': 'Datos yfinance', 'nn_guard': 'Guardia de errores', 'nn_input': 'Capa de entrada', 'nn_split': 'TR / VA / TE',
    'nn_scaler': 'Escalador', 'nn_embedding': 'Embeddings', 'nn_hidden': 'Capa oculta', 'nn_output': 'Capa de salida',
    'nn_train': 'Entrenar', 'nn_finetune': 'Ajuste fino', 'nn_predict': 'Predicción e inferencia',
}
ACTS: dict[str, Any] = {
    'linear': nn.Identity, 'relu': nn.ReLU, 'leaky_relu': lambda: nn.LeakyReLU(0.01), 'gelu': nn.GELU, 'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid, 'swish': nn.SiLU, 'silu': nn.SiLU, 'elu': nn.ELU, 'softmax': lambda: nn.Softmax(dim=-1),
}
OPTS = ('adam', 'adamw', 'sgd', 'rmsprop')
PRICE_COLS = ('Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume')


def _state(frame: pd.DataFrame, need: str | None = None) -> dict[str, Any]:
    return S.state(frame, FAMILY, need, LABEL)


def _emit(frame: pd.DataFrame, st: dict[str, Any], step: str, out: pd.DataFrame | None = None) -> pd.DataFrame:
    return S.emit(frame, FAMILY, st, step, out)


def _log(st: dict[str, Any], step: str, level: str, message: str) -> dict[str, Any]:
    return {**st, 'log': [*st.get('log', []), {'paso': LABEL.get(step, step), 'nivel': level, 'mensaje': message}]}


def _flag(p: dict[str, Any], key: str, default: bool) -> bool:
    v = p.get(key, default)
    return v if isinstance(v, bool) else str(v).strip().lower() in ('1', 'true', 'yes', 'sí', 'si', 'on')


def _num(p: dict[str, Any], key: str, default: float, lo: float, hi: float, what: str) -> float:
    try:
        v = float(p.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{what}: debe ser numérico.') from exc
    if not lo <= v <= hi or not math.isfinite(v):
        raise ValueError(f'{what}: debe estar entre {lo} y {hi} (recibido {v}).')
    return v


def _choice(p: dict[str, Any], key: str, default: str, options: tuple[str, ...], what: str) -> str:
    v = str(p.get(key, default)).strip().lower()
    if v not in options:
        raise ValueError(f'{what}: opciones {", ".join(options)} (recibido «{v}»).')
    return v


def _safe_name(name: str, what: str) -> str:
    name = str(name or '').strip()
    if not name or any(c in name for c in '/\\:') or name.startswith('.') or len(name) > 64:
        raise ValueError(f'{what}: nombre simple sin barras ni puntos iniciales (máx. 64 caracteres).')
    return name


def _opt_spec(p: dict[str, Any], prefix: str = '', inherit: bool = False) -> dict[str, Any] | None:
    """Optimizador de un bloque. Con ``inherit`` y valor «inherit» devuelve None (usa el del bloque anterior)."""
    options = ('inherit', *OPTS) if inherit else OPTS
    name = _choice(p, f'{prefix}optimizer', 'inherit' if inherit else 'adam', options, 'Optimizador')
    if name == 'inherit':
        return None
    return {'optimizer': name, 'lr': _num(p, f'{prefix}lr', 1e-3, 1e-6, 1.0, 'Learning rate'),
            'weight_decay': _num(p, f'{prefix}weight_decay', 0.0, 0.0, 1.0, 'Weight decay'),
            'momentum': _num(p, f'{prefix}momentum', 0.9, 0.0, 0.999, 'Momentum'),
            'clip': _num(p, f'{prefix}clip', 1.0, 0.0, 1000.0, 'Clipping de gradiente (0 = sin clip)')}


# ══ 1 · Datos yfinance ═══════════════════════════════════════════════════════════════════════════════
def _synthetic(ticker: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + sum(map(ord, ticker)))
    vol = 0.012 * np.exp(np.cumsum(rng.normal(0, 0.03, n)).clip(-1, 1))
    ret = 0.0003 + vol * rng.standard_t(5, n) / np.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(ret))
    gap = rng.normal(0, 0.002, n)
    open_ = close / np.exp(ret) * np.exp(gap)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n, name='Date')
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'AdjClose': close,
                         'Volume': rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=idx)


def data(frame: pd.DataFrame | None, p: dict[str, Any], progress: S.Progress = None):
    """yfinance → OHLCV limpio. Reintentos con espera exponencial; si todo falla: caché en disco o serie sintética."""
    tickers = S.names(str(p.get('ticker', 'SPY')).upper()) or ['SPY']
    retries = int(_num(p, 'retries', 3, 0, 10, 'Reintentos'))
    fallback = _choice(p, 'fallback', 'cache', ('none', 'cache', 'synthetic'), 'Fallback')
    source = _choice(p, 'source', 'yfinance', ('yfinance', 'synthetic'), 'Fuente')
    price = _choice(p, 'price_column', 'close', ('close', 'adjclose'), 'Columna de precio')
    log: list[dict[str, str]] = []
    frames, origin = [], {}
    for t in tickers:
        got, err = None, None
        if source == 'synthetic':
            got, origin[t] = _synthetic(t, int(_num(p, 'synthetic_rows', 1500, 200, 20000, 'Filas sintéticas')), int(p.get('seed', 42))), 'sintético'
        else:
            q = {**p, 'ticker': t, 'actions': True}
            for attempt in range(retries + 1):
                try:
                    got, _ = yfinance_data(q, progress)
                    origin[t] = 'yfinance'
                    break
                except ValueError as exc:                        # validación o ticker sin datos: reintentar no ayuda
                    err = exc
                    if 'no devolvió datos' not in str(exc) or attempt == retries:
                        break
                except Exception as exc:                         # red / API: reintento con espera exponencial
                    err = exc
                    if attempt < retries:
                        wait = min(0.5 * 2 ** attempt, 8)
                        log.append({'paso': LABEL['nn_data'], 'nivel': 'aviso', 'mensaje': f'{t}: intento {attempt + 1} falló ({type(exc).__name__}); reintento en {wait:.1f}s.'})
                        if progress: progress({'detail': f'{t}: reintento {attempt + 1}/{retries}'})
                        time.sleep(wait)
            cache = ROOT / 'cache' / f'{t}_{p.get("interval", "1d")}.pkl'
            if got is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                got.to_pickle(cache)
            elif fallback == 'cache' and cache.exists():
                got, origin[t] = pd.read_pickle(cache), f'caché en disco ({time.strftime("%Y-%m-%d", time.localtime(cache.stat().st_mtime))})'
                log.append({'paso': LABEL['nn_data'], 'nivel': 'aviso', 'mensaje': f'{t}: Yahoo falló ({err}); se usa la última copia en caché.'})
            elif fallback == 'synthetic':
                got, origin[t] = _synthetic(t, 1500, int(p.get('seed', 42))), 'sintético (fallback)'
                log.append({'paso': LABEL['nn_data'], 'nivel': 'aviso', 'mensaje': f'{t}: Yahoo falló ({err}); se usa una serie sintética. NO son datos reales.'})
            else:
                raise ValueError(f'{t}: no se pudieron obtener datos de Yahoo ({err}). Revisa el símbolo/conexión o activa un fallback (cache / synthetic).')
        got = got.copy()
        got.index = pd.DatetimeIndex(got.index).tz_localize(None) if pd.DatetimeIndex(got.index).tz is not None else pd.DatetimeIndex(got.index)
        if price == 'adjclose' and 'AdjClose' in got:
            got['Close'] = got['AdjClose']
        got['Ticker'] = t
        frames.append(got)
    lens = {t: len(f) for t, f in zip(tickers, frames)}
    if len(frames) > 1:                                        # multi-ticker: alinear calendarios (intersección de fechas)
        common = frames[0].index
        for f in frames[1:]:
            common = common.intersection(f.index)
        if len(common) < 100:
            raise ValueError(f'Las fechas comunes entre {", ".join(tickers)} son sólo {len(common)}: calendarios desalineados (mezcla de cripto y acciones?).')
        dropped = {t: n - len(common) for t, n in lens.items() if n != len(common)}
        if dropped:
            log.append({'paso': LABEL['nn_data'], 'nivel': 'info', 'mensaje': f'Alineación de calendario: filas descartadas {dropped}.'})
        frames = [f.loc[common] for f in frames]
    out = pd.concat(frames).sort_index(kind='stable') if len(frames) > 1 else frames[0]
    out.index.name = 'Date'
    out.attrs.update({'raw_rows': len(out), 'ticker': tickers[0], 'tickers': tickers})
    st = {'tickers': tickers, 'origin': origin, 'log': log, 'interval': str(p.get('interval', '1d'))}
    rows = [{'fecha': S.day(d), 'Close': float(c)} for d, c in zip(frames[0].index, frames[0]['Close'])][-1200:]
    return _emit(out, st, 'nn_data', out), S.report(
        'nn_data', selected=' · '.join(f'{t}: {origin[t]} ({lens[t]} filas)' for t in tickers),
        metrics={'filas': len(out), 'tickers': len(tickers)}, table=log,
        charts=[S.chart(f'{tickers[0]} · Close', rows, ['Close'], 'fecha')],
        notes=['Precio: ' + ('Adj Close (ajustado por dividendos y splits)' if price == 'adjclose' else 'Close (auto_adjust de yfinance ya ajusta por dividendos y splits)') + '.',
               f'Reintentos: {retries} con espera exponencial (0.5 s, 1 s, 2 s…). Fallback: {fallback}. Caché en disco: artifacts/nn/cache.',
               'Multi-índice de yfinance aplanado y zona horaria eliminada aguas arriba; multi-ticker = intersección de fechas.'])


# ══ 2 · Guardia de errores ═══════════════════════════════════════════════════════════════════════════
def guard(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    """Diagnóstico transversal. Se puede colocar después de cualquier herramienta del submenú."""
    st = _state(frame)
    policy = _choice(p, 'policy', 'repair', ('fail', 'repair', 'warn'), 'Política')
    max_nan = _num(p, 'max_nan_ratio', 0.05, 0.0, 1.0, 'Máximo de faltantes tolerado')
    checks, out = [], frame.copy()
    num = out.select_dtypes(include='number')
    inf = int(np.isinf(num.to_numpy(float)).sum()) if not num.empty else 0
    nan = int(num.isna().to_numpy().sum()) if not num.empty else 0
    ratio = nan / max(num.size, 1)
    dup = int(out.index.duplicated().sum()) if 'Ticker' not in out else int(out.reset_index().duplicated(['Date', 'Ticker']).sum())
    mono = bool(pd.DatetimeIndex(out.index).is_monotonic_increasing) if isinstance(out.index, pd.DatetimeIndex) else False
    bad_px = int(((num[[c for c in ('Open', 'High', 'Low', 'Close') if c in num]] <= 0).any(axis=1)).sum()) if 'Close' in num else 0
    jumps = 0
    if 'Close' in num:
        r = np.log(num['Close'].astype(float)).diff().abs()
        jumps = int((r > _num(p, 'jump_threshold', 0.25, 0.01, 5, 'Umbral de salto')).sum())
    checks += [('Inf', inf, inf == 0), ('NaN', nan, ratio <= max_nan), ('Fechas duplicadas', dup, dup == 0),
               ('Orden cronológico', 'sí' if mono else 'no', mono), ('Precios ≤ 0', bad_px, bad_px == 0),
               (f'Saltos |log r| > umbral', jumps, True)]
    for key in ('X', 'Xs'):                                       # forma y finitud del estado (tras la capa de entrada)
        if key in st:
            arr = st[key]
            checks.append((f'{key} finito {arr.shape}', int((~np.isfinite(arr)).sum()), bool(np.isfinite(arr).all())))
    if 'X' in st and 'y' in st and len(st['X']) != len(st['y']):
        checks.append(('Filas X vs y', f'{len(st["X"])} ≠ {len(st["y"])}', False))
    failed = [c for c in checks if not c[2]]
    actions: list[str] = []
    if failed and policy == 'fail':
        raise ValueError('Guardia: ' + '; '.join(f'{n} = {v}' for n, v, _ in failed) + '. Cambia la política a repair o corrige la fuente.')
    if failed and policy == 'repair':
        if inf:
            out = out.replace([np.inf, -np.inf], np.nan); actions.append('Inf → NaN')
        if dup:
            out = out[~out.reset_index().duplicated(['Date', 'Ticker'] if 'Ticker' in out else ['Date'], keep='last').to_numpy()]; actions.append(f'{dup} duplicados eliminados (se conserva el último)')
        if not mono:
            out = out.sort_index(kind='stable'); actions.append('ordenado por fecha')
        if bad_px:
            out = out[~(out[[c for c in ('Open', 'High', 'Low', 'Close') if c in out]] <= 0).any(axis=1)]; actions.append(f'{bad_px} filas con precio ≤ 0 eliminadas')
        numc = out.select_dtypes(include='number').columns
        if out[numc].isna().any().any():
            before = len(out)
            out[numc] = out.groupby('Ticker')[list(numc)].ffill() if 'Ticker' in out else out[numc].ffill()
            out = out.dropna(subset=[c for c in ('Close',) if c in out])
            actions.append(f'NaN rellenados hacia delante (sin mirar el futuro); {before - len(out)} filas iniciales descartadas')
    for n, v, ok in failed:
        st = _log(st, 'nn_guard', 'error' if policy == 'fail' else 'aviso', f'{n} = {v}')
    for a in actions:
        st = _log(st, 'nn_guard', 'info', f'Reparación: {a}')
    if 'X' in st and not np.isfinite(st['X']).all():
        raise ValueError('Guardia: la matriz de entrada contiene NaN/Inf. Coloca la guardia antes de la capa de entrada o usa nan_policy = drop.')
    table = [{'control': n, 'valor': v, 'estado': 'OK' if ok else ('REPARADO' if policy == 'repair' else 'FALLO')} for n, v, ok in checks]
    return _emit(frame, st, 'nn_guard', out), S.report(
        'nn_guard', selected=f'{len(checks) - len(failed)}/{len(checks)} controles OK · política {policy}',
        metrics={'controles': len(checks), 'fallos': len(failed), 'filas': len(out)}, table=table + [{'control': '— bitácora —', 'valor': '', 'estado': ''}] + [
            {'control': e['paso'], 'valor': e['mensaje'], 'estado': e['nivel']} for e in st.get('log', [])],
        notes=['fail detiene el Blueprint con un diagnóstico; repair corrige sin mirar el futuro (ffill, dedupe, orden); warn sólo registra.',
               'La bitácora reúne los avisos de todas las herramientas nn_* aguas arriba (reintentos, fallbacks, reparaciones).'])


# ══ 3 · Capa de entrada ══════════════════════════════════════════════════════════════════════════════
def _series(frame: pd.DataFrame, col: str) -> pd.Series:
    by = frame.groupby('Ticker', sort=False) if 'Ticker' in frame else None
    close = frame['Close'].astype(float)
    if col == 'LogReturn':
        return by['Close'].transform(lambda s: np.log(s.astype(float)).diff()) if by is not None else np.log(close).diff()
    if col == 'Range':
        return np.log(frame['High'].astype(float) / frame['Low'].astype(float))
    if col == 'LogVolume':
        return np.log1p(frame['Volume'].astype(float))
    if col not in frame:
        raise ValueError(f'Columna «{col}» inexistente. Disponibles: LogReturn, Range, LogVolume, {", ".join(c for c in frame.columns if c != "Ticker")}.')
    return pd.to_numeric(frame[col], errors='coerce').astype(float)


def input_layer(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    """Construye X (n, d) e y (n, h) causales. Modos: simple (1 parámetro), polynomial, lags."""
    st = _state(frame)
    mode = _choice(p, 'mode', 'lags', ('simple', 'polynomial', 'lags'), 'Modo de entrada')
    cols = S.names(p.get('features', 'LogReturn')) or ['LogReturn']
    if mode == 'simple' and len(cols) != 1:
        raise ValueError('Modo simple: exactamente 1 parámetro (p. ej. LogReturn).')
    target = str(p.get('target', 'LogReturn')).strip()
    h = int(_num(p, 'horizon', 1, 1, 30, 'Horizonte'))
    k = int(_num(p, 'lags', 10, 1, 250, 'Ventana de lags'))
    stride = int(_num(p, 'lag_stride', 1, 1, 20, 'Stride de lags'))
    degree = int(_num(p, 'degree', 2, 1, 4, 'Grado polinómico'))
    inter = _flag(p, 'interaction_only', False)
    nan_policy = _choice(p, 'nan_policy', 'drop', ('drop', 'zero', 'ffill'), 'Nulos iniciales')
    act = _choice(p, 'activation', 'linear', tuple(k_ for k_ in ACTS if k_ != 'softmax'), 'Activación de entrada')
    proj = int(_num(p, 'projection_units', 0, 0, 1024, 'Unidades de proyección'))
    opt = _opt_spec(p)
    feats = pd.DataFrame({c: _series(frame, c) for c in cols}, index=frame.index)
    by = frame['Ticker'] if 'Ticker' in frame else pd.Series('_', index=frame.index)
    names: list[str] = []
    if mode == 'lags':
        parts = {}
        for c in cols:
            for lag in range(0, k * stride, stride):            # x_t, x_{t-s}, … : todo conocido al cierre t
                parts[f'{c}_t-{lag}' if lag else f'{c}_t'] = feats[c].groupby(by.to_numpy(), sort=False).shift(lag)
        X = pd.DataFrame(parts)
    elif mode == 'polynomial':
        base = feats
        parts = {c: base[c] for c in cols}
        for d in range(2, degree + 1):
            for combo in combinations_with_replacement(cols, d):
                if inter and len(set(combo)) < len(combo):
                    continue
                parts['·'.join(combo) if len(set(combo)) > 1 else f'{combo[0]}^{d}'] = np.prod([base[c] for c in combo], axis=0)
        X = pd.DataFrame(parts, index=frame.index)
    else:
        X = feats.copy()
    names = list(X.columns)
    tgt = _series(frame, target)
    Y = pd.DataFrame({f'{target}_t+{j}': tgt.groupby(by.to_numpy(), sort=False).shift(-j) for j in range(1, h + 1)})
    lead = int(X.isna().any(axis=1).sum())
    if nan_policy == 'zero':
        X = X.fillna(0.0)
    elif nan_policy == 'ffill':
        X = X.groupby(by.to_numpy(), sort=False).ffill()
    X = X.replace([np.inf, -np.inf], np.nan)
    have_x = X.notna().all(axis=1)
    live = have_x & Y.isna().any(axis=1)                         # rasgos disponibles pero objetivo aún futuro → inferencia en vivo
    ok = have_x & Y.notna().all(axis=1)
    if ok.sum() < 100:
        raise ValueError(f'Sólo {int(ok.sum())} filas útiles tras lags/horizonte: amplía la historia o reduce lags ({k}) / horizonte ({h}).')
    d = len(names)
    if d > 4096:
        raise ValueError(f'{d} rasgos: demasiados términos polinómicos. Baja el grado o usa interaction_only.')
    keep = ok.to_numpy()
    st = {**st, 'X': X[ok].to_numpy(np.float64), 'y': Y[ok].to_numpy(np.float64), 'names': names, 'y_names': list(Y.columns),
          'dates': pd.DatetimeIndex(frame.index[keep]), 'ticker': by[ok].to_numpy(), 'horizon': h, 'target': target,
          'X_live': X[live].to_numpy(np.float64), 'live_dates': pd.DatetimeIndex(frame.index[live.to_numpy()]), 'live_ticker': by[live].to_numpy(),
          'last_price': frame.loc[live, 'Close'].astype(float).to_numpy() if 'Close' in frame else None,
          'input_spec': {'mode': mode, 'features': cols, 'd': d, 'shape': [None, d], 'seq_shape': [None, k, len(cols)] if mode == 'lags' else None,
                         'activation': act, 'projection_units': proj, 'optimizer': opt},
          'layers': [], 'embeddings': None}
    st = _log(st, 'nn_input', 'info', f'{lead} filas iniciales con lags incompletos tratadas con «{nan_policy}».')
    shape = f'(lote, {d})' + (f' ≡ (lote, {k}, {len(cols)}) aplanado' if mode == 'lags' else '')
    return _emit(frame, st, 'nn_input'), S.report(
        'nn_input', selected=f'{mode} · {d} rasgos → {shape} · objetivo {target} t+1..t+{h}', features=names[:40],
        metrics={'filas': int(ok.sum()), 'rasgos': d, 'horizonte': h, 'filas_en_vivo': int(live.sum()), 'nulos_iniciales': lead},
        table=[{'bloque': 'Input', 'forma': shape, 'activación': act, 'proyección': proj or 'identidad',
                'optimizador base': f'{opt["optimizer"]} lr={opt["lr"]} wd={opt["weight_decay"]} clip={opt["clip"]}'}],
        notes=['Lags: x_t, x_{t-s}, …, x_{t-(k-1)s} por ticker (shift causal); objetivo y_{t+1..t+h} = shift(-j).',
               'Polinomios: potencias y productos cruzados hasta el grado elegido (interaction_only omite las potencias puras).',
               'El optimizador de este bloque es el optimizador base: las capas ocultas/salida pueden sobrescribirlo.',
               *(['Activación no lineal sin proyección: se aplica elemento a elemento sobre las entradas (poco habitual).'] if act != 'linear' and not proj else [])])


# ══ 4 · TR / VA / TE ═════════════════════════════════════════════════════════════════════════════════
def split(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_input')
    dates, h = st['dates'], st['horizon']
    tr, va = _num(p, 'train_ratio', 0.7, 0.3, 0.9, 'Fracción TR'), _num(p, 'val_ratio', 0.15, 0.05, 0.4, 'Fracción VA')
    if tr + va >= 0.95:
        raise ValueError('TR + VA debe ser < 0.95 para dejar datos de prueba (TE).')
    gap = int(_num(p, 'gap', 0, 0, 500, 'Hueco adicional'))
    mode = _choice(p, 'mode', 'holdout', ('holdout', 'walk_forward'), 'Modo de validación')
    folds = int(_num(p, 'folds', 4, 2, 10, 'Folds walk-forward'))
    uniq = pd.DatetimeIndex(sorted(set(dates)))
    vs, ts = uniq[int(tr * len(uniq))], uniq[int((tr + va) * len(uniq))]
    pos = uniq.get_indexer(dates)
    purge = h + gap                                              # el objetivo t+1..t+h no puede cruzar el corte
    ivs, its = uniq.get_loc(vs), uniq.get_loc(ts)
    g = np.where(pos < ivs - purge, 0, np.where((pos >= ivs) & (pos < its - purge), 1, np.where(pos >= its, 2, -1)))
    counts = {k: int((g == i).sum()) for i, k in enumerate(('train', 'validation', 'test'))}
    if min(counts.values()) < 30:
        raise ValueError(f'Cada bloque necesita ≥ 30 filas ({counts}). Amplía la historia o mueve los cortes.')
    wf = []
    if mode == 'walk_forward':                                   # ventana expansiva dentro de TR+VA
        pool = np.where(g >= 0)[0][g[g >= 0] < 2]
        edges = np.linspace(len(pool) * 0.4, len(pool), folds + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            wf.append((pool[:max(a - purge, 1)], pool[a:b]))
    st = {**st, 'group': g, 'cuts': {'VAL_START': S.day(vs), 'TEST_START': S.day(ts)}, 'walk_forward': wf}
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    out['Split'] = np.where(idx < vs, 'TR', np.where(idx < ts, 'VA', 'TE'))
    return _emit(frame, st, 'nn_split', out), S.report(
        'nn_split', selected=f'VAL_START {S.day(vs)} · TEST_START {S.day(ts)} · {mode}', splits=counts,
        metrics={**counts, 'purgadas': int((g < 0).sum()), 'folds': len(wf)},
        table=[{'bloque': k, 'filas': v} for k, v in counts.items()] + [{'bloque': f'purgadas (h={h} + hueco {gap})', 'filas': int((g < 0).sum())}]
              + [{'bloque': f'walk-forward fold {i + 1}', 'filas': f'TR {len(a)} → VA {len(b)}'} for i, (a, b) in enumerate(wf)],
        notes=['Orden cronológico sin barajar; con varios tickers el corte es por fecha (mismo instante para todos).',
               f'Purga: se eliminan las {purge} filas anteriores a cada corte para que ninguna etiqueta TR/VA vea el bloque siguiente.',
               'Walk-forward: ventana expansiva; nn_train entrena un modelo por fold y reporta la media en VA.'])


# ══ 5 · Escalador ════════════════════════════════════════════════════════════════════════════════════
def _make_scaler(method: str, lo: float, hi: float):
    return {'standard': StandardScaler, 'minmax': lambda: MinMaxScaler((lo, hi)), 'robust': RobustScaler, 'maxabs': MaxAbsScaler,
            'none': lambda: None}[method]()


def scaler(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_split')
    op = _choice(p, 'operation', 'fit_transform', ('fit_transform', 'load_transform'), 'Operación')
    name = _safe_name(p.get('save_as', 'scaler'), 'Nombre del escalador') if (op == 'load_transform' or _flag(p, 'persist', False)) else None
    path = ROOT / 'scalers' / f'{name}.joblib' if name else None
    if op == 'load_transform':
        if not path.exists():
            raise ValueError(f'No existe el escalador guardado «{name}» ({path}). Ejecuta antes un Blueprint con persist = true.')
        blob = joblib.load(path)
        if blob['names'] != st['names']:
            raise ValueError(f'El escalador «{name}» se ajustó con otros rasgos ({len(blob["names"])} vs {len(st["names"])}): la capa de entrada debe coincidir.')
        sx, sy, method = blob['x'], blob['y'], blob['method']
    else:
        method = _choice(p, 'method', 'standard', ('standard', 'minmax', 'robust', 'maxabs', 'none'), 'Método')
        lo, hi = (-1.0, 1.0) if str(p.get('feature_range', '0,1')).replace(' ', '') == '-1,1' else (0.0, 1.0)
        tr = st['group'] == 0
        sx = _make_scaler(method, lo, hi)
        sy = _make_scaler(method, lo, hi) if _flag(p, 'scale_target', True) else None
        if sx is not None:
            sx.fit(st['X'][tr])                                  # fit SÓLO con TR
        if sy is not None:
            sy.fit(st['y'][tr])
    Xs = sx.transform(st['X']) if sx is not None else st['X'].copy()
    ys = sy.transform(st['y']) if sy is not None else st['y'].copy()
    Xl = sx.transform(st['X_live']) if sx is not None and len(st['X_live']) else st['X_live']
    if path is not None and op == 'fit_transform':
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'x': sx, 'y': sy, 'method': method, 'names': st['names']}, path)
    const = int((np.nanstd(st['X'][st['group'] == 0], axis=0) == 0).sum())
    st = {**st, 'Xs': Xs.astype(np.float32), 'ys': ys.astype(np.float32), 'Xs_live': np.asarray(Xl, np.float32), 'x_scaler': sx, 'y_scaler': sy,
          'scaler_method': method}
    if const:
        st = _log(st, 'nn_scaler', 'aviso', f'{const} rasgos constantes en TR (desviación 0): no aportan información.')
    g = st['group']
    table = [{'bloque': b, 'media X': float(Xs[g == i].mean()), 'σ X': float(Xs[g == i].std()), 'media y': float(ys[g == i].mean())}
             for i, b in enumerate(('TR', 'VA', 'TE'))]
    return _emit(frame, st, 'nn_scaler'), S.report(
        'nn_scaler', selected=f'{method} · {op}' + (f' · {path.name}' if path else ''), splits=S.splits(st), table=table,
        metrics={'rasgos': Xs.shape[1], 'constantes_en_TR': const},
        notes=['fit sólo con TR → transform en TR/VA/TE/en vivo → inverse_transform en nn_predict (unidades reales).',
               'VA y TE no tienen media 0 / σ 1 exactas: es la señal de que no hubo fuga.',
               'persist = true guarda {x, y, método, rasgos} con joblib en artifacts/nn/scalers para inferencia o ajuste fino.'])


# ══ 6 · Embeddings ═══════════════════════════════════════════════════════════════════════════════════
def embedding(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_scaler')
    dim = int(_num(p, 'dim', 4, 1, 64, 'Dimensión'))
    fuse = _choice(p, 'fusion', 'concat', ('concat', 'add'), 'Fusión')
    fields: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    dates, ldates = st['dates'], st['live_dates']
    if _flag(p, 'weekday', True):
        fields['día semana'] = (dates.dayofweek.to_numpy(), ldates.dayofweek.to_numpy(), 7)
    if _flag(p, 'month', True):
        fields['mes'] = ((dates.month - 1).to_numpy(), (ldates.month - 1).to_numpy(), 12)
    if _flag(p, 'regime', True):                                # terciles de volatilidad realizada; umbrales sólo con TR
        vol = pd.Series(st['X'][:, 0]).groupby(st['ticker']).transform(lambda s: s.abs().rolling(20, min_periods=5).mean()).bfill().to_numpy()
        lvol = np.abs(st['X_live'][:, 0]) if len(st['X_live']) else np.zeros(0)
        cuts = np.quantile(vol[st['group'] == 0], [1 / 3, 2 / 3])
        fields['régimen vol'] = (np.digitize(vol, cuts), np.digitize(lvol, cuts), 3)
    if _flag(p, 'ticker', True) and len(set(st['ticker'])) > 1:
        cats = {t: i for i, t in enumerate(sorted(set(st['ticker'])))}
        fields['ticker'] = (np.array([cats[t] for t in st['ticker']]), np.array([cats.get(t, 0) for t in st['live_ticker']]), len(cats))
    if not fields:
        raise ValueError('Activa al menos un embedding (día de la semana, mes, régimen o ticker).')
    spec = {'names': list(fields), 'cards': [c for _, _, c in fields.values()], 'dim': dim, 'fusion': fuse, 'freeze': _flag(p, 'freeze', False)}
    st = {**st, 'cat': np.stack([a for a, _, _ in fields.values()], 1).astype(np.int64),
          'cat_live': np.stack([b for _, b, _ in fields.values()], 1).astype(np.int64) if len(ldates) else np.zeros((0, len(fields)), np.int64),
          'embeddings': spec}
    extra = dim * len(fields) if fuse == 'concat' else 0
    return _emit(frame, st, 'nn_embedding'), S.report(
        'nn_embedding', selected=f'{", ".join(fields)} · dim {dim} · {fuse}',
        table=[{'campo': n, 'categorías': c, 'dimensión': dim, 'parámetros': c * dim} for n, (_, _, c) in fields.items()],
        metrics={'campos': len(fields), 'dimensión_añadida': extra},
        notes=['nn.Embedding(categorías, dim) por campo; se aprende junto con la red (o queda congelado con freeze).',
               'concat: [x_numérico, e_1, …, e_m] entra a la primera capa; add: cada embedding se proyecta a d y se suma a x.',
               'Los umbrales del régimen de volatilidad se calculan sólo con TR (sin fuga).'])


# ══ 7 · Capa oculta (duplicable) ═════════════════════════════════════════════════════════════════════
def hidden(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_input')
    size = str(p.get('units', '64'))
    units = int(_num(p, 'custom_units', 64, 1, 4096, 'Unidades personalizadas')) if size == 'custom' else int(size)
    if size != 'custom' and units not in (32, 64, 128, 256):
        raise ValueError('Unidades: 32, 64, 128, 256 o custom.')
    prev = st['layers'][-1] if st['layers'] else None
    act = _choice(p, 'activation', 'relu', ('inherit', *[a for a in ACTS if a != 'softmax']), 'Activación')
    if act == 'inherit':
        act = prev['activation'] if prev else 'relu'
    norm = _choice(p, 'norm', 'none', ('none', 'batch', 'layer'), 'Normalización')
    base = {'units': units, 'activation': act, 'dropout': _num(p, 'dropout', 0.0, 0.0, 0.9, 'Dropout'), 'norm': norm,
            'l1': _num(p, 'l1', 0.0, 0.0, 1.0, 'L1'), 'l2': _num(p, 'l2', 0.0, 0.0, 1.0, 'L2'), 'residual': _flag(p, 'residual', False),
            'optimizer': _opt_spec(p, inherit=True), 'freeze': _flag(p, 'freeze', False)}
    repeat = int(_num(p, 'repeat', 1, 1, 8, 'Repeticiones'))
    name = str(p.get('name') or 'Oculta').strip()[:40]
    layers = list(st['layers'])
    for r in range(repeat):                                     # duplicar = misma configuración, pesos independientes
        layers.append({**deepcopy(base), 'name': f'{name} {len(layers) + 1}' if repeat > 1 or not p.get('name') else name})
    st = {**st, 'layers': layers}
    d = st['input_spec']['projection_units'] or st['input_spec']['d']
    rows = []
    for L in layers:
        params = d * L['units'] + L['units']
        rows.append({'capa': L['name'], 'forma': f'({d}) → ({L["units"]})', 'activación': L['activation'], 'dropout': L['dropout'], 'norm': L['norm'],
                     'L1/L2': f'{L["l1"]}/{L["l2"]}', 'residual': ('sí' if d == L['units'] else 'proyección') if L['residual'] else 'no',
                     'optimizador': (L['optimizer'] or {}).get('optimizer', 'hereda'), 'parámetros': params})
        d = L['units']
    return _emit(frame, st, 'nn_hidden'), S.report(
        'nn_hidden', selected=f'{name}: {repeat} × Dense({units}, {act}) · total {len(layers)} capas ocultas', table=rows,
        metrics={'capas_ocultas': len(layers), 'unidades': units},
        notes=['Duplicar: copia el nodo (Ctrl+C / Ctrl+V) y encadénalo, o usa «Repeticiones» para N capas idénticas con pesos independientes.',
               'activation = inherit toma la activación de la capa anterior; optimizer = inherit usa el optimizador base de la capa de entrada.',
               'Orden del bloque: Linear → Norm → activación → Dropout (+ atajo residual; proyección lineal si cambian las unidades).',
               'Un optimizador propio crea un grupo de parámetros con su lr / weight decay (o un optimizador distinto para esta capa).'])


# ══ 8 · Capa de salida ═══════════════════════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, d_in: int, L: dict[str, Any]):
        super().__init__()
        self.linear = nn.Linear(d_in, L['units'])
        self.norm = nn.BatchNorm1d(L['units']) if L['norm'] == 'batch' else nn.LayerNorm(L['units']) if L['norm'] == 'layer' else nn.Identity()
        self.act = ACTS[L['activation']]()
        self.drop = nn.Dropout(L['dropout'])
        self.skip = (nn.Identity() if d_in == L['units'] else nn.Linear(d_in, L['units'], bias=False)) if L['residual'] else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm(self.linear(x))))
        return y + self.skip(x) if self.skip is not None else y


class Net(nn.Module):
    def __init__(self, spec: dict[str, Any]):
        super().__init__()
        self.spec = spec
        d = spec['input']['d']
        emb = spec.get('embeddings')
        self.emb = nn.ModuleList(nn.Embedding(c, emb['dim']) for c in emb['cards']) if emb else nn.ModuleList()
        self.emb_proj = nn.ModuleList(nn.Linear(emb['dim'], d, bias=False) for _ in emb['cards']) if emb and emb['fusion'] == 'add' else None
        if emb and emb['fusion'] == 'concat':
            d += emb['dim'] * len(emb['cards'])
        proj = spec['input']['projection_units']
        self.inp = nn.Sequential(nn.Linear(d, proj), ACTS[spec['input']['activation']]()) if proj else ACTS[spec['input']['activation']]()
        d = proj or d
        self.hidden = nn.ModuleList()
        for L in spec['layers']:
            self.hidden.append(Block(d, L))
            d = L['units']
        self.out = nn.Linear(d, spec['output']['units'])

    def forward(self, x: torch.Tensor, cat: torch.Tensor | None = None) -> torch.Tensor:
        if len(self.emb):
            es = [e(cat[:, i]) for i, e in enumerate(self.emb)]
            x = x + sum(pj(e) for pj, e in zip(self.emb_proj, es)) if self.emb_proj is not None else torch.cat([x, *es], 1)
        x = self.inp(x)
        for b in self.hidden:
            x = b(x)
        return self.out(x)                                       # logits / valor lineal; la activación de salida se aplica en predict


LOSSES = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'huber': nn.HuberLoss, 'bce': nn.BCEWithLogitsLoss, 'crossentropy': nn.CrossEntropyLoss}


def output_layer(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_scaler')
    task = _choice(p, 'task', 'regression', ('regression', 'direction', 'multiclass'), 'Tarea')
    h = st['horizon']
    act = _choice(p, 'activation', {'regression': 'linear', 'direction': 'sigmoid', 'multiclass': 'softmax'}[task], ('linear', 'sigmoid', 'softmax', 'tanh'), 'Activación de salida')
    loss = _choice(p, 'loss', {'regression': 'mse', 'direction': 'bce', 'multiclass': 'crossentropy'}[task], tuple(LOSSES), 'Pérdida')
    valid = {'regression': ('mse', 'mae', 'huber'), 'direction': ('bce',), 'multiclass': ('crossentropy',)}[task]
    if loss not in valid:
        raise ValueError(f'Pérdida «{loss}» incompatible con la tarea {task}: usa {", ".join(valid)}.')
    if task == 'regression' and act in ('sigmoid', 'softmax'):
        raise ValueError(f'Regresión con activación {act}: acota la salida a (0,1) y el objetivo escalado no lo está. Usa linear (o tanh con MinMax [-1,1]).')
    if task == 'direction' and act != 'sigmoid':
        raise ValueError('Dirección (binaria): activación sigmoid (la pérdida BCE usa logits).')
    if task == 'multiclass' and (act != 'softmax' or h != 1):
        raise ValueError('Multiclase (baja / lateral / sube): activación softmax y horizonte 1 en la capa de entrada.')
    if act == 'tanh' and st['scaler_method'] != 'minmax':
        st = _log(st, 'nn_output', 'aviso', 'tanh acota a [-1,1]: conviene MinMax [-1,1] en el escalador del objetivo.')
    units = 3 if task == 'multiclass' else h
    opt = _opt_spec(p, inherit=True)
    metrics = S.names(p.get('metrics', 'rmse,mae,r2,directional_accuracy')) if task == 'regression' else ['accuracy', 'f1', 'auc'] if task == 'direction' else ['accuracy', 'f1_macro']
    out = {'task': task, 'units': units, 'activation': act, 'loss': loss, 'huber_delta': _num(p, 'huber_delta', 1.0, 0.01, 100, 'Delta de Huber'),
           'flat_band': _num(p, 'flat_band', 0.002, 0.0, 0.1, 'Banda lateral'), 'metrics': metrics, 'optimizer': opt,
           'postprocess': _choice(p, 'postprocess', 'none', ('none', 'clip_quantile', 'sign'), 'Post-procesado')}
    spec = {'input': st['input_spec'], 'layers': st['layers'], 'embeddings': st.get('embeddings'), 'output': out}
    torch.manual_seed(int(p.get('seed', 42)))
    try:
        net = Net(spec)
        with torch.no_grad():
            xb = torch.as_tensor(st['Xs'][:4])
            cb = torch.as_tensor(st['cat'][:4]) if spec['embeddings'] else None
            net.eval()
            y = net(xb, cb)
    except RuntimeError as exc:
        raise ValueError(f'Shapes incompatibles entre capas: {exc}') from exc
    n_params = sum(v.numel() for v in net.parameters())
    st = {**st, 'spec': spec, 'net': net}
    arch = ' → '.join([f'Input({st["input_spec"]["d"]})', *[f'{L["units"]}·{L["activation"]}' for L in st['layers']], f'Output({units}·{act})'])
    return _emit(frame, st, 'nn_output'), S.report(
        'nn_output', selected=arch, metrics={'parámetros': n_params, 'unidades_salida': units, 'capas_ocultas': len(st['layers'])},
        table=[{'tarea': task, 'unidades': units, 'activación': act, 'pérdida': loss, 'métricas': ', '.join(metrics),
                'optimizador': (opt or {}).get('optimizer', 'hereda'), 'post-procesado': out['postprocess'], 'forma': f'{tuple(y.shape)}'}],
        notes=['Compilación: comprobación de shapes con un lote de prueba (forward en seco) antes de entrenar.',
               'BCE/CrossEntropy reciben logits (estable numéricamente); sigmoid/softmax se aplican al predecir.',
               'Regresión multi-horizonte: una unidad por paso t+1..t+h (salida directa, sin recursión).'])


# ══ 9 · Entrenar ═════════════════════════════════════════════════════════════════════════════════════
def _targets(st: dict[str, Any]) -> np.ndarray:
    task, out = st['spec']['output']['task'], st['spec']['output']
    raw = st['y']
    if task == 'direction':
        return (raw > 0).astype(np.float32)
    if task == 'multiclass':
        b = out['flat_band']
        return np.where(raw[:, 0] > b, 2, np.where(raw[:, 0] < -b, 0, 1)).astype(np.int64)
    return st['ys']


def _optimizers(net: Net, spec: dict[str, Any]) -> list[torch.optim.Optimizer]:
    base = spec['input']['optimizer']
    groups: dict[str, list[dict[str, Any]]] = {}
    def add(params, opt):
        o = opt or base
        params = [q for q in params if q.requires_grad]
        if params:
            groups.setdefault(o['optimizer'], []).append({'params': params, 'lr': o['lr'], 'weight_decay': o['weight_decay'], 'momentum': o['momentum']})
    add([*net.inp.parameters(), *net.emb.parameters(), *(net.emb_proj.parameters() if net.emb_proj is not None else [])], None)
    for b, L in zip(net.hidden, spec['layers']):
        add(b.parameters(), L['optimizer'])
    add(net.out.parameters(), spec['output']['optimizer'])
    made = []
    for name, gs in groups.items():
        if name in ('sgd', 'rmsprop'):
            made.append((torch.optim.SGD if name == 'sgd' else torch.optim.RMSprop)(gs))
        else:
            for g in gs: g.pop('momentum')
            made.append((torch.optim.AdamW if name == 'adamw' else torch.optim.Adam)(gs))
    return made


def _penalty(net: Net, spec: dict[str, Any]) -> torch.Tensor:
    pen = torch.zeros(())
    for b, L in zip(net.hidden, spec['layers']):
        w = b.linear.weight
        if L['l1']: pen = pen + L['l1'] * w.abs().sum()
        if L['l2']: pen = pen + L['l2'] * (w ** 2).sum()
    return pen


def _loss_fn(spec: dict[str, Any]):
    o = spec['output']
    return nn.HuberLoss(delta=o['huber_delta']) if o['loss'] == 'huber' else LOSSES[o['loss']]()


def _fit(net: Net, st: dict[str, Any], p: dict[str, Any], tr: np.ndarray, va: np.ndarray, progress: S.Progress, tag: str = ''):
    spec = st['spec']
    dev = S.device(p)
    epochs, bs = int(_num(p, 'epochs', 60, 1, 2000, 'Épocas')), int(_num(p, 'batch_size', 64, 4, 8192, 'Lote'))
    patience, min_delta = int(_num(p, 'patience', 10, 1, 500, 'Paciencia')), _num(p, 'min_delta', 0.0, 0.0, 1.0, 'Mejora mínima')
    nan_policy = _choice(p, 'nan_policy', 'reduce_lr', ('reduce_lr', 'stop'), 'Ante NaN')
    sched_kind = _choice(p, 'scheduler', 'plateau', ('none', 'plateau', 'cosine'), 'Scheduler')
    clip = max([spec['input']['optimizer']['clip'], *[(L['optimizer'] or {}).get('clip', 0) for L in spec['layers']]])
    X, Y = torch.as_tensor(st['Xs']), torch.as_tensor(_targets(st))
    C = torch.as_tensor(st['cat']) if spec.get('embeddings') else None
    net = net.to(dev)
    opts = _optimizers(net, spec)
    scheds = [torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=max(patience // 3, 1)) if sched_kind == 'plateau'
              else torch.optim.lr_scheduler.CosineAnnealingLR(o, epochs) if sched_kind == 'cosine' else None for o in opts]
    lossf = _loss_fn(spec)
    best, best_state, best_ep, wait, hist, events = math.inf, deepcopy(net.state_dict()), 0, 0, [], []
    gen = torch.Generator().manual_seed(int(p.get('seed', 42)))
    tri, vai = torch.as_tensor(tr), torch.as_tensor(va)
    for ep in range(1, epochs + 1):
        net.train()
        perm = tri[torch.randperm(len(tri), generator=gen)]
        tot, nb, bad = 0.0, 0, False
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and any(L['norm'] == 'batch' for L in spec['layers']):
                continue
            xb, yb = X[idx].to(dev), Y[idx].to(dev)
            pred = net(xb, C[idx].to(dev) if C is not None else None)
            loss = lossf(pred, yb) + _penalty(net, spec).to(dev)
            if not torch.isfinite(loss):
                bad = True
                break
            for o in opts: o.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), clip if clip > 0 else float('inf'))
            if not torch.isfinite(gn):
                bad = True
                break
            for o in opts: o.step()
            tot, nb = tot + loss.item(), nb + 1
        if bad:                                                   # divergencia: volver al mejor checkpoint y bajar el lr, o parar
            if nan_policy == 'stop' or len(events) >= 3:
                events.append(f'época {ep}: pérdida/gradiente NaN → parada en el mejor checkpoint (época {best_ep}).')
                break
            net.load_state_dict(best_state)
            for o in opts:
                for g in o.param_groups: g['lr'] *= 0.5
            events.append(f'época {ep}: pérdida/gradiente NaN → se restaura el checkpoint y lr × 0.5.')
            continue
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(X[vai].to(dev), C[vai].to(dev) if C is not None else None), Y[vai].to(dev)))
        hist.append({'epoch': ep, 'train': tot / max(nb, 1), 'validation': vl})
        for s in scheds:
            if isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau): s.step(vl)
            elif s is not None: s.step()
        if progress: progress({'epoch': ep, 'epochs': epochs, 'detail': f'{tag}época {ep}/{epochs} · VA {vl:.4f}'})
        if vl < best - min_delta:
            best, best_state, best_ep, wait = vl, deepcopy(net.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= patience:
                events.append(f'early stopping en la época {ep} (sin mejora en {patience} épocas).')
                break
    net.load_state_dict(best_state)
    return net.cpu().eval(), hist, best, best_ep, events


def _save(st: dict[str, Any], net: Net, name: str) -> Path:
    path = ROOT / 'models' / f'{name}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'spec': st['spec'], 'state': net.state_dict(), 'names': st['names'], 'x_scaler': st['x_scaler'], 'y_scaler': st['y_scaler']}, path)
    return path


def train(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_output')
    g = st['group']
    tr, va = np.where(g == 0)[0], np.where(g == 1)[0]
    table, t0 = [], time.perf_counter()
    if st.get('walk_forward'):
        for i, (a, b) in enumerate(st['walk_forward']):
            torch.manual_seed(int(p.get('seed', 42)))
            _, _, fb, fe, _ = _fit(Net(st['spec']), st, p, a, b, progress, f'fold {i + 1} · ')
            table.append({'fold': i + 1, 'TR': len(a), 'VA': len(b), 'mejor pérdida VA': fb, 'época': fe})
    torch.manual_seed(int(p.get('seed', 42)))
    net = Net(st['spec'])
    net.load_state_dict(st['net'].state_dict())
    net, hist, best, ep, events = _fit(net, st, p, tr, va, progress)
    if not hist:
        raise ValueError('El entrenamiento divergió desde la primera época (NaN). Baja el learning rate, activa clipping o revisa el escalado.')
    saved = _save(st, net, _safe_name(p.get('save_as'), 'Guardar como')) if str(p.get('save_as') or '').strip() else None
    st = {**st, 'net': net, 'history': hist, 'best_epoch': ep}
    for e in events:
        st = _log(st, 'nn_train', 'aviso' if 'NaN' in e else 'info', e)
    metrics = {'mejor_pérdida_VA': best, 'mejor_época': ep, 'épocas': len(hist), 'segundos': time.perf_counter() - t0}
    if table:
        metrics['VA_walk_forward_media'] = float(np.mean([r['mejor pérdida VA'] for r in table]))
    return _emit(frame, st, 'nn_train'), S.report(
        'nn_train', selected=f'checkpoint época {ep} · pérdida VA {best:.5f}' + (f' · guardado {saved.name}' if saved else ''), splits=S.splits(st),
        metrics=metrics, table=table or [{'evento': e} for e in events],
        charts=[S.chart('Pérdida train / validación', hist, ['train', 'validation'], 'epoch')],
        notes=['zero_grad → forward → pérdida + L1/L2 → backward → clip_grad_norm_ → step (un optimizador por tipo, grupos por capa).',
               'Checkpoint = época con menor pérdida en VA; early stopping con paciencia y mejora mínima.',
               'Ante NaN/Inf en pérdida o gradiente: se restaura el mejor checkpoint y se reduce el lr a la mitad (máx. 3 veces).',
               *(['Walk-forward: un modelo nuevo por fold (ventana expansiva); el modelo final se entrena en TR completo.'] if table else []),
               *(([f'Modelo + escaladores guardados en {saved}.']) if saved else [])])


# ══ 10 · Ajuste fino ═════════════════════════════════════════════════════════════════════════════════
def finetune(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_scaler')
    pre = p.pop('_pretrained', None)
    src = _choice(p, 'source', 'pin', ('pin', 'file'), 'Origen del modelo base')
    if src == 'pin':
        if not isinstance(pre, pd.DataFrame):
            raise ValueError('Conecta la salida de «Entrenar» del modelo base al pin pretrained (o usa source = file).')
        base = _state(pre, 'nn_train')
        spec, state, names, origin = base['spec'], base['net'].state_dict(), base['names'], 'pin pretrained'
    else:
        name = _safe_name(p.get('load', ''), 'Modelo a cargar')
        path = ROOT / 'models' / f'{name}.pt'
        if not path.exists():
            raise ValueError(f'No existe el modelo «{name}» ({path}). Entrena uno con «Guardar como».')
        blob = torch.load(path, weights_only=False)
        spec, state, names, origin = blob['spec'], blob['state'], blob['names'], path.name
    if len(names) != len(st['names']):
        raise ValueError(f'El modelo base espera {len(names)} rasgos y esta capa de entrada produce {len(st["names"])}: usa la misma configuración de entrada.')
    if spec.get('embeddings') and 'cat' not in st:
        raise ValueError('El modelo base usa embeddings: añade «Embeddings» con los mismos campos antes del ajuste fino.')
    if spec.get('embeddings') and st['embeddings']['cards'] != spec['embeddings']['cards']:
        raise ValueError(f'Embeddings incompatibles: base {spec["embeddings"]["cards"]} vs actual {st["embeddings"]["cards"]} (p. ej. distinto número de tickers).')
    freeze = _choice(p, 'freeze', 'hidden_bottom', ('none', 'hidden_bottom', 'all_but_output', 'embeddings'), 'Congelar')
    n_freeze = int(_num(p, 'freeze_layers', 1, 0, 32, 'Capas a congelar'))
    factor = _num(p, 'lr_factor', 0.1, 0.001, 1.0, 'Factor de learning rate')
    spec = deepcopy(spec)
    for o in [spec['input']['optimizer'], *[L['optimizer'] for L in spec['layers']], spec['output']['optimizer']]:
        if o: o['lr'] *= factor
    net = Net(spec)
    net.load_state_dict(state)
    frozen = []
    if freeze == 'all_but_output':
        for n_, q in net.named_parameters():
            if not n_.startswith('out.'): q.requires_grad = False
        frozen = ['todo salvo la salida']
    elif freeze == 'hidden_bottom':
        for q in [*net.inp.parameters(), *net.emb.parameters()]: q.requires_grad = False
        for b, L in list(zip(net.hidden, spec['layers']))[:n_freeze]:
            for q in b.parameters(): q.requires_grad = False
            frozen.append(L['name'])
        frozen.insert(0, 'entrada/embeddings')
    elif freeze == 'embeddings':
        for q in net.emb.parameters(): q.requires_grad = False
        frozen = ['embeddings']
    st2 = {**st, 'spec': spec, 'net': net}
    # pérdida del modelo base SIN ajustar en VA (referencia)
    va = np.where(st['group'] == 1)[0]
    lossf = _loss_fn(spec)
    with torch.no_grad():
        C = torch.as_tensor(st['cat'][va]) if spec.get('embeddings') else None
        before = float(lossf(net.eval()(torch.as_tensor(st['Xs'][va]), C), torch.as_tensor(_targets(st2)[va])))
    net, hist, best, ep, events = _fit(net, st2, {'patience': 8, **p}, np.where(st['group'] == 0)[0], va, progress, 'ajuste fino · ')
    if not hist:
        raise ValueError('El ajuste fino divergió (NaN): baja lr_factor.')
    saved = _save(st2, net, _safe_name(p.get('save_as'), 'Guardar como')) if str(p.get('save_as') or '').strip() else None
    st2 = {**st2, 'net': net, 'history': hist, 'best_epoch': ep}
    trainable = sum(q.numel() for q in net.parameters() if q.requires_grad)
    return _emit(frame, st2, 'nn_train'), S.report(
        'nn_finetune', selected=f'base {origin} · congelado: {", ".join(frozen) or "nada"} · lr × {factor}', splits=S.splits(st2),
        metrics={'pérdida_VA_base_sin_ajuste': before, 'pérdida_VA_ajustada': best, 'mejora': before - best, 'parámetros_entrenables': trainable,
                 'parámetros_totales': sum(q.numel() for q in net.parameters()), 'mejor_época': ep},
        table=[{'evento': e} for e in events], charts=[S.chart('Ajuste fino · pérdida train / validación', hist, ['train', 'validation'], 'epoch')],
        notes=['Se reutilizan arquitectura y pesos del modelo base; el escalador es el de esta rama (o cárgalo con load_transform).',
               'hidden_bottom congela entrada/embeddings y las N primeras capas ocultas; all_but_output sólo reentrena la salida.',
               'Si la pérdida ajustada no mejora a la del modelo base, el checkpoint conserva los pesos base (época 0 implícita).',
               *([f'Guardado en {saved}.'] if saved else [])])


# ══ 11 · Predicción e inferencia ═════════════════════════════════════════════════════════════════════
def _metrics_reg(y: np.ndarray, yh: np.ndarray) -> dict[str, float]:
    e = y - yh
    ss = ((y - y.mean()) ** 2).sum()
    return {'RMSE': float(np.sqrt((e ** 2).mean())), 'MAE': float(np.abs(e).mean()), 'R2': float(1 - (e ** 2).sum() / ss) if ss else 0.0,
            'acierto_dirección': float((np.sign(y) == np.sign(yh)).mean())}


def predict(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'nn_train')
    net, spec = st['net'].eval(), st['spec']
    o = spec['output']
    alpha = _num(p, 'interval', 0.9, 0.5, 0.99, 'Nivel de la banda')
    with torch.no_grad():
        C = torch.as_tensor(st['cat']) if spec.get('embeddings') else None
        raw = net(torch.as_tensor(st['Xs']), C).numpy()
        Cl = torch.as_tensor(st['cat_live']) if spec.get('embeddings') and len(st['Xs_live']) else None
        raw_live = net(torch.as_tensor(st['Xs_live']), Cl).numpy() if len(st['Xs_live']) else np.zeros((0, raw.shape[1]))
    g, dates = st['group'], st['dates']
    table, charts, metrics, out_rows = [], [], {}, None
    if o['task'] == 'regression':
        sy = st['y_scaler']
        f = (lambda a: sy.inverse_transform(a) if sy is not None and len(a) else a)
        if o['activation'] == 'tanh':
            raw, raw_live = np.tanh(raw), np.tanh(raw_live)
        yh, yh_live, y = f(raw), f(raw_live), st['y']
        if o['postprocess'] == 'clip_quantile':
            lo, hi = np.quantile(y[g == 0], [0.01, 0.99], axis=0)
            yh, yh_live = np.clip(yh, lo, hi), np.clip(yh_live, lo, hi)
        q = np.quantile(np.abs(y[g == 1] - yh[g == 1]), alpha, axis=0)   # banda conformal: residuos de VA
        for i, b in enumerate(('TR', 'VA', 'TE')):
            m = _metrics_reg(y[g == i, 0], yh[g == i, 0])
            naive = _metrics_reg(y[g == i, 0], np.zeros_like(y[g == i, 0]))['RMSE']
            cover = float((np.abs(y[g == i] - yh[g == i]) <= q).mean())
            table.append({'bloque': b, **m, 'RMSE ingenuo (0)': naive, f'cobertura banda {alpha:.0%}': cover})
        metrics = {f'TE_{k}': v for k, v in table[2].items() if k != 'bloque'}
        te = np.where(g == 2)[0]
        rows = [{'fecha': S.day(dates[i]), 'real': float(y[i, 0]), 'predicción': float(yh[i, 0]), 'inferior': float(yh[i, 0] - q[0]),
                 'superior': float(yh[i, 0] + q[0])} for i in te[S.thin(len(te))]]
        charts.append({**S.chart(f'TE · {st["y_names"][0]} real vs predicción (banda {alpha:.0%})', rows, ['real', 'predicción', 'inferior', 'superior'], 'fecha')})
        live = [{'fecha_señal': S.day(d), 'ticker': t, **{n: float(v) for n, v in zip(st['y_names'], r)},
                 **({'precio_estimado_t+h': float(px * math.exp(r.sum()))} if st['target'] == 'LogReturn' and st['last_price'] is not None else {}),
                 'banda ±': float(q[0])} for d, t, r, px in zip(st['live_dates'], st['live_ticker'], yh_live,
                                                              st['last_price'] if st['last_price'] is not None else [0] * len(yh_live))]
        out_rows = pd.DataFrame({'Pred': yh[:, 0], 'Lower': yh[:, 0] - q[0], 'Upper': yh[:, 0] + q[0]}, index=dates)
    else:
        prob = 1 / (1 + np.exp(-raw)) if o['task'] == 'direction' else np.exp(raw - raw.max(1, keepdims=True)) / np.exp(raw - raw.max(1, keepdims=True)).sum(1, keepdims=True)
        y = _targets(st)
        pred = (prob > 0.5).astype(int) if o['task'] == 'direction' else prob.argmax(1)
        for i, b in enumerate(('TR', 'VA', 'TE')):
            yy, pp = (y[g == i, 0], pred[g == i, 0]) if o['task'] == 'direction' else (y[g == i], pred[g == i])
            table.append({'bloque': b, 'accuracy': float((yy == pp).mean()), 'clase mayoritaria': float(max(np.bincount(yy.astype(int)) / len(yy)))})
        metrics = {f'TE_{k}': v for k, v in table[2].items() if k != 'bloque'}
        lp = 1 / (1 + np.exp(-raw_live)) if o['task'] == 'direction' else np.exp(raw_live) / np.exp(raw_live).sum(1, keepdims=True) if len(raw_live) else raw_live
        live = [{'fecha_señal': S.day(d), 'ticker': t, **({f'P(sube t+{j + 1})': float(v) for j, v in enumerate(r)} if o['task'] == 'direction'
                 else {'P(baja)': float(r[0]), 'P(lateral)': float(r[1]), 'P(sube)': float(r[2])})} for d, t, r in zip(st['live_dates'], st['live_ticker'], lp)]
        out_rows = pd.DataFrame({'Pred': prob[:, 0] if o['task'] == 'direction' else pred}, index=dates)
    exported = None
    if str(p.get('export') or '').strip():
        exported = ROOT / 'exports' / f'{_safe_name(p.get("export"), "Exportar como")}.csv'
        exported.parent.mkdir(parents=True, exist_ok=True)
        e = out_rows.copy()
        e['Split'] = np.select([g == 0, g == 1, g == 2], ['TR', 'VA', 'TE'], 'purga')
        e['Ticker'] = st['ticker']
        e.to_csv(exported)
    out = frame.copy()
    if 'Ticker' not in out and not out_rows.index.duplicated().any():
        out = out.join(out_rows, how='left')
    return _emit(frame, st, 'nn_predict', out), S.report(
        'nn_predict', selected=f'{o["task"]} · {len(live)} predicción(es) en vivo' + (f' · exportado {exported.name}' if exported else ''), splits=S.splits(st),
        metrics=metrics, table=table + [{'bloque': '— en vivo (últimos datos, objetivo aún desconocido) —'}] + live, charts=charts,
        notes=['TR/VA/TE con la misma red (checkpoint de VA); TE es la única medida fuera de muestra.',
               'Regresión: inverse_transform del escalador devuelve las unidades reales; banda = cuantil de |residuo| en VA (conformal split).',
               'En vivo: filas cuyos rasgos ya existen pero cuyo objetivo aún es futuro (las últimas h barras).',
               *([f'CSV exportado en {exported}.'] if exported else [])])


STEPS = {'nn_guard': guard, 'nn_input': input_layer, 'nn_split': split, 'nn_scaler': scaler, 'nn_embedding': embedding,
         'nn_hidden': hidden, 'nn_output': output_layer, 'nn_train': train, 'nn_finetune': finetune, 'nn_predict': predict}
SOURCES = {'nn_data': data}
