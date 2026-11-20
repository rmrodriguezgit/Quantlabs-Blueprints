"""Laya-MLX typed decisions on market data: ask, calibrate (weights), learn online (feedback).

Three independent nodes, each a learning step:

1. ``laya_signal``    — asks Laya three typed questions per day using only data known at t.
2. ``laya_calibrate`` — fits combination weights over Laya's answers on TR, selects on VA, scores TE.
3. ``laya_feedback``  — walk-forward: predicts, waits for the realized return, then updates weights.

Laya's own transformer weights stay frozen; what is learned here is a small logistic layer
on top of its calibrated probabilities. No future observation enters a prediction.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

Progress = Callable[[dict[str, Any]], None] | None

CHECKPOINTS = {'laya-mlx': 'aac6fef/laya-mlx', 'laya-multilingual-mlx': 'aac6fef/laya-multilingual-mlx'}
FEATURES = ['bias', 'z_trend', 'z_higher', 'risk']
_AGENTS: dict[str, Any] = {}
_CACHE: dict[str, dict[str, Any]] = {}


def _agent(name: str) -> Any:
    if name not in CHECKPOINTS:
        raise ValueError(f'Checkpoint must be one of: {", ".join(CHECKPOINTS)}.')
    if name not in _AGENTS:
        try:
            import laya_mlx
        except ImportError as exc:
            raise ValueError('laya-mlx no está instalado. Ejecuta: .venv/bin/pip install laya-mlx (requiere Apple Silicon y macOS 14+).') from exc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _AGENTS[name] = laya_mlx.load(CHECKPOINTS[name])
    return _AGENTS[name]


def _questions(p: dict[str, Any]) -> dict[str, Any]:
    return {
        'trend': {'type': 'choice', 'instructions': str(p.get('trend_question', 'Given these technical indicators, what is the most likely direction of the next trading day?')),
                  'criteria': {'up': 'the price rises tomorrow', 'down': 'the price falls tomorrow', 'flat': 'the price barely changes tomorrow'}},
        'risk': {'type': 'score', 'instructions': str(p.get('risk_question', 'How risky is holding this asset tomorrow?')),
                 'criteria': ['low risk', 'medium risk', 'high risk']},
        'higher': {'type': 'noul', 'instructions': str(p.get('higher_question', 'The asset will close higher on the next trading day.'))},
    }


def _states(frame: pd.DataFrame, ticker: str) -> list[dict[str, Any]]:
    """One state per row, built only from values available at the close of that row."""
    close = frame['Close'].astype(float)
    log = np.log(close / close.shift(1))
    view = pd.DataFrame(index=frame.index)
    view['return_1d_pct'] = log * 100
    view['return_5d_pct'] = log.rolling(5).sum() * 100
    view['volatility_20d_annual_pct'] = log.rolling(20).std() * np.sqrt(252) * 100
    if 'RSI' in frame: view['rsi_14'] = frame['RSI']
    if 'MACDHistogram' in frame: view['macd_histogram'] = frame['MACDHistogram']
    if 'SMA' in frame: view['close_vs_sma_pct'] = (close / frame['SMA'] - 1) * 100
    if 'EMA' in frame: view['close_vs_ema_pct'] = (close / frame['EMA'] - 1) * 100
    if {'BBUpper', 'BBLower'} <= set(frame.columns):
        width = (frame['BBUpper'] - frame['BBLower']).replace(0, np.nan)
        view['bollinger_position_0_to_1'] = (close - frame['BBLower']) / width
    view = view.replace([np.inf, -np.inf], np.nan)
    return [{'asset': ticker, **{k: (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}} for row in view.to_dict('records')]


def laya_signal(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if 'Close' not in frame or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Laya Signal requiere filas fechadas con Close (Market Data → Technical Features).')
    max_rows = int(p.get('max_rows', 500))
    if not 30 <= max_rows <= 3000:
        raise ValueError('Máximo de filas: 30–3000.')
    ticker = str(p.get('ticker', 'ASSET'))
    data = frame.iloc[-max_rows:].copy()
    states = _states(data, ticker)
    questions = _questions(p)
    agent = _agent(str(p.get('checkpoint', 'laya-mlx')))
    qkey = json.dumps(questions, sort_keys=True) + str(p.get('checkpoint'))
    started = time.perf_counter()
    keys = [qkey + json.dumps(state, sort_keys=True) for state in states]
    todo = [i for i, k in enumerate(keys) if k not in _CACHE]
    hits = len(states) - len(todo)
    chunk = 64
    for j in range(0, len(todo), chunk):   # batched forward passes share the question schema
        part = todo[j:j + chunk]
        results = agent.predict_batch([states[i] for i in part], questions, batch_size=chunk) if hasattr(agent, 'predict_batch') else [agent.predict(states[i], questions) for i in part]
        for i, res in zip(part, results):
            _CACHE[keys[i]] = res['answers']
        if progress:
            progress({'detail': f'Laya {min(j + chunk, len(todo))}/{len(todo)} días'})
    answers = [_CACHE[k] for k in keys]
    elapsed = time.perf_counter() - started
    data['Laya_P_up'] = [a['trend']['probabilities']['up'] for a in answers]
    data['Laya_P_down'] = [a['trend']['probabilities']['down'] for a in answers]
    data['Laya_P_flat'] = [a['trend']['probabilities']['flat'] for a in answers]
    data['Laya_P_higher'] = [a['higher']['noul'] for a in answers]
    data['Laya_Risk'] = [a['risk']['score'] for a in answers]
    data['Laya_Confidence'] = [a['trend']['answer_confidence'] for a in answers]
    choice = pd.Series([a['trend']['choice'] for a in answers], index=data.index)
    data.attrs['raw_rows'] = frame.attrs.get('raw_rows', len(frame))
    data.attrs['laya_ticker'] = ticker
    nxt = np.sign(np.log(data['Close'].shift(-1) / data['Close']))
    agree = ((choice == 'up') & (nxt > 0)) | ((choice == 'down') & (nxt < 0))
    scored = choice.isin(['up', 'down']) & nxt.notna() & (nxt != 0)
    counts = choice.value_counts().to_dict()
    sample = states[-1]
    rows = [{'date': d.strftime('%Y-%m-%d'), 'P_up': float(u), 'P_down': float(dn), 'P_higher': float(h)}
            for d, u, dn, h in zip(data.index, data['Laya_P_up'], data['Laya_P_down'], data['Laya_P_higher'])]
    report = {
        'topic': 'laya_signal', 'selected': str(p.get('checkpoint', 'laya-mlx')), 'features': list(sample.keys()),
        'splits': {'train': 0, 'validation': 0, 'test': len(data)},
        'notes': [
            'Laya no genera texto: hace una sola pasada bidireccional y devuelve probabilidades calibradas por pregunta.',
            'Estado del día t = indicadores conocidos al cierre de t. La respuesta se compara con el cierre de t+1.',
            f'Preguntas: choice (up/down/flat), score (riesgo 0–2), noul (P de cerrar más alto). {len(states) - hits} inferencias nuevas, {hits} desde caché, {elapsed:.1f}s.',
            'Los pesos del transformer de Laya están congelados; el aprendizaje ocurre en Laya Calibrate y Laya Feedback.',
            'Último estado enviado: ' + json.dumps(sample, ensure_ascii=False),
        ],
        'table': [{'respuesta': k, 'días': int(v), 'proporción': round(v / len(choice), 4)} for k, v in counts.items()],
        'metrics': {'dias': len(data), 'acierto_crudo_up_down': float(agree[scored].mean()) if scored.any() else None,
                    'dias_evaluables': int(scored.sum()), 'ms_por_dia': elapsed / max(len(states) - hits, 1) * 1000 if hits < len(states) else None,
                    'confianza_media': float(data['Laya_Confidence'].mean())},
        'charts': [
            {'title': 'Probabilidades de Laya por día', 'rows': rows, 'series': ['P_up', 'P_down', 'P_higher'], 'x_key': 'date', 'chart_type': 'line'},
            {'title': 'Distribución de P(up)', 'rows': [{'P_up': float(v)} for v in data['Laya_P_up']], 'series': ['P_up'], 'x_key': 'P_up', 'chart_type': 'histogram'},
        ],
    }
    return data, report


# ---------- shared helpers for the learned layer ----------

def _design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    need = {'Laya_P_up', 'Laya_P_down', 'Laya_P_higher', 'Laya_Risk', 'Close'}
    if not need <= set(frame.columns):
        raise ValueError('Conecta primero Laya Signal: faltan columnas ' + ', '.join(sorted(need - set(frame.columns))) + '.')
    eps = 1e-4
    up, down = frame['Laya_P_up'].clip(eps, 1), frame['Laya_P_down'].clip(eps, 1)
    higher = frame['Laya_P_higher'].clip(eps, 1 - eps)
    x = np.column_stack([np.ones(len(frame)), np.log(up / down), np.log(higher / (1 - higher)), frame['Laya_Risk'] - 1]).astype(float)
    r_next = np.log(frame['Close'].shift(-1) / frame['Close']).to_numpy(dtype=float)
    y = (r_next > 0).astype(float)
    return x, y, r_next


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0, 1, bins + 1)
    rows, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append({'bin': round(float((lo + hi) / 2), 2), 'predicha': float(p[m].mean()), 'observada': float(y[m].mean()), 'n': int(m.sum())})
    return float(total), rows


def _scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {'accuracy': float(((p > .5) == (y > .5)).mean()), 'brier': float(np.mean((p - y) ** 2)), 'logloss': _logloss(p, y), 'ece': _ece(p, y)[0]}


RAW = np.array([0.0, 1.0, 0.0, 0.0])  # "trust Laya's up/down odds as is"


def laya_calibrate(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    tr, va = float(p.get('train_ratio', .6)), float(p.get('val_ratio', .2))
    lr, epochs, l2 = float(p.get('lr', .1)), int(p.get('epochs', 300)), float(p.get('l2', .01))
    if not (.3 <= tr <= .8 and .05 <= va <= .3 and tr + va < .95):
        raise ValueError('Fracciones: TR 0.3–0.8, VA 0.05–0.3 y TR+VA < 0.95.')
    if not (1e-4 <= lr <= 5 and 10 <= epochs <= 5000 and 0 <= l2 <= 10):
        raise ValueError('lr 0.0001–5, épocas 10–5000, L2 0–10.')
    x, y, r_next = _design(frame)
    x, y, r_next, dates = x[:-1], y[:-1], r_next[:-1], frame.index[:-1]
    n = len(y)
    a, b = int(n * tr), int(n * (tr + va))
    if a < 20 or b - a < 10 or n - b < 10:
        raise ValueError('Muy pocas filas para TR/VA/TE; aumenta máx. filas en Laya Signal.')
    mu, sd = x[:a, 1:].mean(0), x[:a, 1:].std(0) + 1e-8          # scaler fitted on TR only
    xs = x.copy(); xs[:, 1:] = (x[:, 1:] - mu) / sd
    w = np.array([mu[0], sd[0], 0.0, 0.0])  # σ(w·xs) == Laya cruda at epoch 0
    history, best = [], (np.inf, w.copy(), 0)
    for epoch in range(1, epochs + 1):
        pt = _sigmoid(xs[:a] @ w)
        grad = xs[:a].T @ (pt - y[:a]) / a + l2 * np.r_[0, w[1:]]
        w -= lr * grad
        lt, lv = _logloss(_sigmoid(xs[:a] @ w), y[:a]), _logloss(_sigmoid(xs[a:b] @ w), y[a:b])
        if lv < best[0]: best = (lv, w.copy(), epoch)
        if epoch % max(1, epochs // 100) == 0 or epoch == 1:
            history.append({'epoch': epoch, 'train': lt, 'validation': lv})
            if progress: progress({'epoch': epoch, 'epochs': epochs, 'train_loss': lt, 'val_loss': lv})
    w = best[1]
    raw_all, cal_all = _sigmoid(x @ RAW), _sigmoid(xs @ w)
    raw_te, cal_te = _scores(raw_all[b:], y[b:]), _scores(cal_all[b:], y[b:])
    base = float(y[:a].mean())
    _, rel_raw = _ece(raw_all[b:], y[b:]); _, rel_cal = _ece(cal_all[b:], y[b:])
    rel = {r['bin']: {'bin': r['bin'], 'perfecta': r['bin'], 'Laya cruda': r['observada']} for r in rel_raw}
    for r in rel_cal: rel.setdefault(r['bin'], {'bin': r['bin'], 'perfecta': r['bin']})['Calibrada'] = r['observada']
    # weights back in original units so they can be read and reused by Laya Feedback
    w_orig = np.r_[w[0] - np.sum(w[1:] * mu / sd), w[1:] / sd]
    out = frame.copy()
    out['Laya_P_raw'] = np.r_[raw_all, np.nan]
    out['Laya_P_cal'] = np.r_[cal_all, np.nan]
    out.attrs.update(frame.attrs)
    out.attrs['laya_weights'] = w_orig.tolist()
    out.attrs['laya_test_start'] = str(dates[b].date())
    names = {'bias': 'sesgo (tendencia base)', 'z_trend': 'log(P_up/P_down)', 'z_higher': 'logit P(higher)', 'risk': 'riesgo − 1'}
    report = {
        'topic': 'laya_calibrate', 'selected': f'mejor época VA = {best[2]}', 'features': FEATURES,
        'splits': {'train': a, 'validation': b - a, 'test': n - b},
        'notes': [
            'Ajuste de pesos: regresión logística p = σ(w·x) sobre las respuestas de Laya. Escalador y gradiente sólo con TR.',
            'Cada época: predecir TR → pérdida log → gradiente → w ← w − lr·∇. VA elige la época (parada temprana); TE se mide una vez.',
            'Pesos iniciales = Laya cruda (sólo log(P_up/P_down)). Si un peso tiende a 0, esa respuesta de Laya no aporta.',
            f'Tasa base TR de días alcistas: {base:.3f}. Un acierto cerca de esa tasa equivale a no tener señal.',
            'Brier y logloss menores = mejor. ECE mide calibración: |probabilidad predicha − frecuencia observada|.',
        ],
        'table': [{'peso': names[f], 'inicial (cruda)': float(RAW[i]), 'ajustado': round(float(w_orig[i]), 4)} for i, f in enumerate(FEATURES)]
                 + [{'peso': f'TE {k}', 'inicial (cruda)': round(raw_te[k], 4), 'ajustado': round(cal_te[k], 4)} for k in raw_te],
        'metrics': {**{f'te_{k}_cruda': v for k, v in raw_te.items()}, **{f'te_{k}_calibrada': v for k, v in cal_te.items()}, 'tasa_base_tr': base},
        'charts': [
            {'title': 'Curva de aprendizaje (logloss)', 'rows': history, 'series': ['train', 'validation'], 'x_key': 'epoch', 'chart_type': 'line'},
            {'title': 'Diagrama de fiabilidad TE', 'rows': sorted(rel.values(), key=lambda r: r['bin']), 'series': ['perfecta', 'Laya cruda', 'Calibrada'], 'x_key': 'bin', 'chart_type': 'scatter_line'},
        ],
    }
    return out, report


def laya_feedback(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    lr, gate, cost = float(p.get('lr', .01)), float(p.get('min_confidence', .52)), float(p.get('cost_bps', 5)) / 1e4
    mode, window = str(p.get('positions', 'long_short')), int(p.get('rolling_window', 20))
    if not (0 <= lr <= 2 and .5 <= gate < 1 and 0 <= cost <= .01 and mode in ('long_short', 'long_only') and 5 <= window <= 120):
        raise ValueError('lr 0–2, confianza mínima 0.5–0.99, coste 0–100 bps, posiciones long_short/long_only, ventana 5–120.')
    x, y, r_next = _design(frame)
    x, y, r_next, dates = x[:-1], y[:-1], r_next[:-1], frame.index[:-1]
    start = frame.attrs.get('laya_test_start')
    i0 = int(np.searchsorted(dates.values, np.datetime64(start))) if start else int(len(y) * float(p.get('start_fraction', .6)))
    if len(y) - i0 < 15:
        raise ValueError('Muy pocos días para la retroalimentación.')
    w = np.array(frame.attrs.get('laya_weights', RAW.tolist()), dtype=float)
    origin = 'Laya Calibrate' if 'laya_weights' in frame.attrs else 'Laya cruda (sin calibrar)'
    w0 = w.copy()
    rows, weights, errors, prev = [], [], [], 0.0
    for t in range(i0, len(y)):
        pt = float(_sigmoid(x[t] @ w))                      # 1) decide with what is known at t
        conf = max(pt, 1 - pt)
        pos = 0.0 if conf < gate else (1.0 if pt > .5 else (-1.0 if mode == 'long_short' else 0.0))
        ret = pos * float(np.expm1(r_next[t])) - cost * abs(pos - prev)   # 2) market reveals t+1
        loss = _logloss(np.array([pt]), np.array([y[t]]))
        if pos != 0 and np.sign(pos) != (1 if y[t] else -1):
            errors.append({'fecha': dates[t].strftime('%Y-%m-%d'), 'P(sube)': round(pt, 4), 'posición': pos, 'retorno real %': round(float(r_next[t]) * 100, 3), 'logloss': round(loss, 4)})
        w = w - lr * (pt - y[t]) * x[t]                      # 3) feedback: learn from the error
        prev = pos
        rows.append({'date': dates[t].strftime('%Y-%m-%d'), 'p': pt, 'y': y[t], 'pos': pos, 'ret': ret, 'mkt': float(np.expm1(r_next[t])), 'loss': loss})
        weights.append({'date': rows[-1]['date'], **{f: float(v) for f, v in zip(FEATURES, w)}})
    df = pd.DataFrame(rows)
    df['equity'] = (1 + df['ret']).cumprod(); df['buy_hold'] = (1 + df['mkt']).cumprod()
    active = df['pos'] != 0
    df['hit'] = np.where(active, (np.sign(df['pos']) == np.where(df['y'] > 0, 1, -1)).astype(float), np.nan)
    df['hit_rate'] = df['hit'].rolling(window, min_periods=5).mean()
    df['rolling_logloss'] = df['loss'].rolling(window, min_periods=5).mean()
    half = len(df) // 2
    sharpe = float(df['ret'].mean() / df['ret'].std() * np.sqrt(252)) if df['ret'].std() > 0 else None
    dd = float((df['equity'] / df['equity'].cummax() - 1).min())
    out = frame.iloc[i0:len(frame) - 1].copy()
    for col, src in (('Feedback_P', 'p'), ('Position', 'pos'), ('StrategyReturn', 'ret'), ('Equity', 'equity'), ('BuyHold', 'buy_hold'), ('RollingHitRate', 'hit_rate')):
        out[col] = df[src].to_numpy()
    out.attrs.update(frame.attrs)
    errors.sort(key=lambda e: -e['logloss'])
    report = {
        'topic': 'laya_feedback', 'selected': f'inicio: {origin}', 'features': FEATURES,
        'splits': {'train': 0, 'validation': 0, 'test': len(df)},
        'notes': [
            'Bucle de retroalimentación por día: 1) predecir con w_t, 2) el mercado revela el retorno t+1, 3) w ← w − lr·(p − y)·x.',
            'El peso se actualiza DESPUÉS de conocer el resultado, nunca antes: no hay mirada al futuro.',
            f'Compuerta de confianza: si max(p, 1−p) < {gate} el agente se abstiene (efectivo). Coste {cost * 1e4:.1f} bps por cambio de posición.',
            'Errores: días con posición en sentido contrario al mercado. La tabla muestra los de mayor logloss.',
            'Compara logloss 1ª mitad vs 2ª mitad: si baja, la retroalimentación está aprendiendo; si sube, el régimen cambió o lr es alto.',
        ],
        'table': [{'peso': f, 'inicio': round(float(a), 4), 'final': round(float(b), 4), 'cambio': round(float(b - a), 4)} for f, a, b in zip(FEATURES, w0, w)] + errors[:10],
        'metrics': {
            'dias': len(df), 'acierto': float(df['hit'].mean()) if active.any() else None, 'abstencion': float(1 - active.mean()),
            'retorno_estrategia': float(df['equity'].iloc[-1] - 1), 'retorno_buy_hold': float(df['buy_hold'].iloc[-1] - 1),
            'sharpe': sharpe, 'max_drawdown': dd, 'errores': len(errors),
            'logloss_1a_mitad': float(df['loss'][:half].mean()), 'logloss_2a_mitad': float(df['loss'][half:].mean()),
        },
        'charts': [
            {'title': 'Capital: estrategia Laya vs comprar y mantener', 'rows': df[['date', 'equity', 'buy_hold']].to_dict('records'), 'series': ['equity', 'buy_hold'], 'x_key': 'date', 'chart_type': 'line'},
            {'title': 'Evolución de los pesos (retroalimentación)', 'rows': weights, 'series': FEATURES, 'x_key': 'date', 'chart_type': 'line'},
            {'title': f'Acierto y logloss móviles ({window} días)', 'rows': df[['date', 'hit_rate', 'rolling_logloss']].dropna().to_dict('records'), 'series': ['hit_rate', 'rolling_logloss'], 'x_key': 'date', 'chart_type': 'line'},
        ],
    }
    return out, report
