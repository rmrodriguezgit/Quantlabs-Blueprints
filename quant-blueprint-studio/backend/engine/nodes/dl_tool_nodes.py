"""Blueprint adapter for the independent DL tools (backend/dl_tools).

The engine never touches the models: it turns a frame column into the tool's typed payload, runs the
tool inside a `with` block (so it is always unloaded) and turns the typed outputs back into columns and
a report. Tools are discovered through the registry, so adding a tool package needs no engine change
beyond this mapping.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from ...dl_tools.core import RunContext, ToolError, registry
    from ...dl_tools.core.layers import parse_layers
except ImportError:  # server started as a script (absolute imports)
    from dl_tools.core import RunContext, ToolError, registry  # type: ignore[no-redef]
    from dl_tools.core.layers import parse_layers  # type: ignore[no-redef]
from .dl_nodes import NetworkSpec

# the infographic's 11 indicators per session (also the Inspector default)
RECURRENT_FEATURES = 'LogReturn,Lag_1,Lag_2,Lag_3,Lag_4,Lag_5,RSI,MACD,MACDSignal,MACDHistogram,Volatility'
ACT = {'ReLU': 'relu', 'Swish': 'swish', 'Tanh': 'tanh', 'GELU': 'gelu', 'Sigmoid': 'sigmoid', 'LeakyReLU': 'leaky_relu', 'Identity': 'identity'}


def network_layers(spec: NetworkSpec) -> list[dict[str, Any]]:
    """Architecture nodes (Network Input → Layers → Network Output) → hidden LayerSpecs.

    Dense makes a layer; a standalone activation, Dropout or BatchNorm modifies the previous Dense.
    The tool keeps its own single input and output layer; the chain only defines what is in between.
    """
    if not spec.complete:
        raise ValueError('La cadena de arquitectura debe terminar en «Network Output».')
    out: list[dict[str, Any]] = []
    for layer in spec.layers:
        kind = str(layer.get('layer_type', 'Dense'))
        if kind == 'Dense':
            out.append({'units': int(layer.get('units', 64)), 'activation': ACT.get(str(layer.get('activation', 'ReLU')), 'relu'), 'dropout': 0.0, 'batch_norm': False})
        elif not out:
            raise ValueError(f'«{kind}» necesita una capa Dense antes en la cadena de arquitectura.')
        elif kind == 'Dropout':
            out[-1]['dropout'] = float(layer.get('rate', 0.2))
        elif kind == 'BatchNorm':
            out[-1]['batch_norm'] = True
        elif kind in ACT:
            out[-1]['activation'] = ACT[kind]
    if len(out) > 8:
        raise ValueError('Máximo 8 capas Dense intermedias.')
    return out

Progress = Callable[[dict[str, Any]], None] | None
KINDS = ('dl_tabular_forecast', 'dl_sequence_classifier', 'dl_anomaly_autoencoder', 'dl_recurrent_compare')


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f'Columna «{name}» inexistente. Disponibles: {", ".join(map(str, frame.columns))}')
    s = pd.to_numeric(frame[name], errors='coerce').astype(float)
    if s.isna().any() or np.isinf(s).any():
        raise ValueError(f'«{name}» tiene faltantes o infinitos: conecta «3 · Limpieza de datos» o Rendimientos con drop_na.')
    return s


def _x(i: Any) -> str:
    return i.isoformat() if hasattr(i, 'isoformat') else str(i)


def _tool_params(kind: str, p: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Keep only the keys the tool's Params schema declares and resolve where the hidden layers come from."""
    allowed = registry.get(kind).Params.model_fields
    out = {k: v for k, v in p.items() if k in allowed and k != 'layers'}
    if isinstance(p.get('_network'), NetworkSpec):
        out['layers'] = network_layers(p['_network'])
        return out, 'cadena Network Input → Network Output conectada'
    text = str(p.get('layers', '') or '').strip()
    if text:
        try:
            out['layers'] = [l.model_dump() for l in parse_layers(text)]
        except (ValueError, TypeError) as exc:
            raise ValueError(f'«Capas a medida» no válidas ({text}): usa unidades:activación:dropout:bn, p. ej. 64:relu:0.1, 32:tanh. {exc}') from exc
        return out, f'capas a medida «{text}»'
    return out, 'nº de capas × neuronas del Inspector'


def recurrent_inputs(frame: pd.DataFrame, p: dict[str, Any]) -> tuple[list[str], str, pd.Series]:
    """Indicators (F columns) and the target of the NEXT session, aligned on the current session's row."""
    mode = str(p.get('target', 'next_intraday'))
    if mode == 'next_intraday':
        o, c = _col(frame, 'Open'), _col(frame, 'Close')
        current, name = c / o - 1, 'retorno intradía (Close/Open − 1)'
    elif mode == 'next_close':
        c = _col(frame, 'Close')
        current, name = np.log(c / c.shift(1)), 'retorno log cierre a cierre'
    else:
        current, name = _col(frame, mode), mode
    target = current.shift(-1)                      # row t holds the value of session t+1
    cols = [c.strip() for c in str(p.get('features', RECURRENT_FEATURES)).split(',') if c.strip()] or \
           [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in ('Open', 'High', 'Low', 'Close', 'Volume', 'Split')]
    return cols, name, target


def architecture_line(rows: list[dict[str, Any]]) -> str:
    """«Entrada LSTMInput → Linear 64 · BatchNorm · ReLU · Dropout 0.1 → Linear 16 · Tanh → Salida Linear 1»."""
    names = {'input_layer': 'Entrada', 'output_layer': 'Salida', 'bottleneck': 'Cuello'}
    parts: list[str] = []
    for r in rows:
        mod, det = str(r['módulo']), str(r['detalle'])
        if mod == 'Linear':
            width = det.split('out_features=')[1].split(',')[0]
            label = f'{names.get(str(r["capa"]), "")} Linear {width}'.strip()
            parts.append(label)
        elif mod.endswith('Input'):
            parts.append(f'Entrada {mod.replace("Input", "")}')
        elif parts:
            extra = {'Dropout': f'Dropout {det.split("p=")[1].split(",")[0]}' if 'p=' in det else 'Dropout', 'BatchNorm1d': 'BatchNorm'}.get(mod, mod)
            parts[-1] += f' · {extra}'
    return ' → '.join(parts)


def run_dl_tool(kind: str, frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    cls = registry.get(kind)
    ctx = RunContext(seed=int(p.get('seed', 42)), device=str(p.get('device', 'auto')))  # type: ignore[arg-type]
    out = frame.copy()
    try:
        params, layer_source = _tool_params(kind, p)
        with cls(params) as tool:
            if progress:
                progress({'detail': f'{cls.spec.title}: cargando y ajustando…'})
            if kind == 'dl_recurrent_compare':
                cols, target_name, target = recurrent_inputs(frame, p)
                params['candidates'] = [n for n, k in (('LSTM', 'use_lstm'), ('GRU', 'use_gru'), ('LSTM (bi)', 'use_lstm_bi'), ('GRU (bi)', 'use_gru_bi')) if p.get(k, k != 'use_gru_bi')]
                tool.params = tool.Params.model_validate(params)
                matrix = pd.concat([_col(frame, c) for c in cols], axis=1)
                result = tool.execute({'features': matrix.to_numpy().tolist(), 'target': target.tolist()}, ctx)
            elif kind == 'dl_anomaly_autoencoder':
                cols = [c.strip() for c in str(p.get('columns', '')).split(',') if c.strip()] or \
                       [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in ('Split',)]
                matrix = pd.concat([_col(frame, c) for c in cols], axis=1)
                result = tool.execute({'data': matrix.to_numpy().tolist()}, ctx)
            else:
                column = str(p.get('column', 'LogRet_Close'))
                result = tool.execute({'series': _col(frame, column).tolist()}, ctx)
    except ToolError as exc:
        raise ValueError(exc.message + (f' · {exc.details.get("errors")}' if exc.details.get('errors') else '')) from exc
    o, d = result.outputs, result.diagnostics
    meta = {'Inferencia (ms)': result.inference_ms}  # device and weights go to the notes (metrics are numeric)
    if kind == 'dl_tabular_forecast':
        h = len(o['forecast'])
        lookback = int(tool.params.lookback)
        start = d['fit_windows'] + lookback        # index of the first holdout target
        resid = np.asarray(o['residuals'])[:, 0]
        actual = _col(frame, column).to_numpy()
        idx = frame.index[start:start + len(resid)]
        out['DL_Forecast_1'] = np.nan
        out.loc[idx, 'DL_Forecast_1'] = actual[start:start + len(resid)] - resid
        rows = [{'x': _x(i), 'real': float(a), 'pronóstico': float(a - r)} for i, a, r in zip(idx, actual[start:start + len(resid)], resid)]
        table = [{'paso': f't+{k + 1}', 'pronóstico': o['forecast'][k], 'inferior': o['lower'][k], 'superior': o['upper'][k]} for k in range(h)]
        metrics = {**meta, 'RMSE holdout': d['holdout_rmse'], 'RMSE línea base': d['naive_rmse'], 'Confianza': o['confidence'], 'Parámetros': d['parameters']}
        notes = [f'{d["architecture"].upper()} · lookback {lookback} · horizonte {h}. Ajuste en las primeras {d["fit_windows"]} ventanas; métricas en {d["holdout_windows"]} ventanas finales no vistas.',
                 f'Confianza = 1 − RMSE/RMSE de la mejor línea base ({d["baseline"]}) = {o["confidence"]:.3f}: ' + ('mejora a la línea base.' if o['confidence'] > 0 else 'no mejora a la línea base: el modelo no aporta.'),
                 f'Próximo valor de {column}: {o["forecast"][0]:.5g} (intervalo {o["lower"][0]:.4g} … {o["upper"][0]:.4g}).']
        inside = float(np.mean(np.abs(resid) <= (o['upper'][0] - o['forecast'][0])))
        notes.append(f'Acierto de dirección en holdout: {d["direction_hit"]:.1%}. Cobertura del intervalo: {inside:.0%} (objetivo {d["coverage_target"]:.0%}).')
        metrics['Acierto dirección'] = d['direction_hit']
        charts = [{'title': f'{column}: real vs pronóstico a 1 paso (holdout)', 'x_key': 'x', 'series': ['real', 'pronóstico'], 'rows': rows, 'chart_type': 'line'}]
    elif kind == 'dl_sequence_classifier':
        names = list(o['probabilities'])
        hist = np.asarray(o['history_labels'])
        lookback = int(tool.params.lookback)
        out['DL_Class'] = np.nan
        out.loc[frame.index[lookback - 1:], 'DL_Class'] = hist
        table = [{'clase': k, 'probabilidad': v} for k, v in o['probabilities'].items()]
        metrics = {**meta, 'Exactitud holdout': o['accuracy'], 'Línea base (mayoritaria)': d['majority_baseline']}
        edge = o['accuracy'] - d['majority_baseline']
        notes = [f'Siguiente paso de {column}: «{o["label"]}» ({", ".join(f"{k} {v:.1%}" for k, v in o["probabilities"].items())}); umbral {d["threshold"]:.2f}.',
                 f'Exactitud en {d["holdout_windows"]} ventanas holdout: {o["accuracy"]:.1%} frente a {d["majority_baseline"]:.1%} de la clase mayoritaria ({edge:+.1%}).',
                 'Sólo es útil si supera a la línea base fuera de muestra.' if edge <= 0 else 'Supera a la línea base fuera de muestra.',
                 'Recall por clase (holdout): ' + ', '.join(f'{k} {v:.0%}' for k, v in d['recall'].items() if v is not None) + '.']
        dominant = max(d['predicted_share'].values())
        if dominant > 0.9:
            notes.append(f'Aviso: predice la misma clase en el {dominant:.0%} de las ventanas; el modelo apenas discrimina (prueba ponderar clases, más épocas o otra señal).')
        charts = [{'title': 'Distribución de la predicción actual', 'x_key': 'clase', 'series': ['probabilidad'], 'rows': table, 'chart_type': 'bar'},
                  {'title': 'Embedding latente de la última ventana', 'x_key': 'dim', 'series': ['valor'],
                   'rows': [{'dim': f'z{i}', 'valor': float(v)} for i, v in enumerate(o['latent'])], 'chart_type': 'bar'}]
        del names
    elif kind == 'dl_recurrent_compare':
        te = d['test']
        unit = 'bps' if float(p.get('target_scale', 10000)) == 10000 else f'× {p.get("target_scale")}'
        rows_te = d['te_rows']
        out['RNN_Pred'] = np.nan
        out.loc[frame.index[rows_te], 'RNN_Pred'] = np.asarray(o['prediction']) / float(p.get('target_scale', 10000))
        table = o['comparison']
        metrics = {**meta, 'MSE TE seleccionado': te['selected_MSE'], 'MSE TE Ridge': te['ridge_MSE'], 'MSE TE predecir 0': te['zero_MSE'],
                   'Acierto dirección': d['hit_selected'], 'Acierto Ridge': d['hit_ridge']}
        if o['next'] is not None:
            metrics['Próxima sesión'] = o['next']
        b, f = d['shape'][1], d['shape'][2]
        gain = 1 - te['selected_MSE'] / te['ridge_MSE'] if te['ridge_MSE'] else 0
        notes = [f'Objetivo: {target_name} de la sesión siguiente, en {unit}. Entradas: {f} indicadores × {b} sesiones ({", ".join(cols)}).',
                 f'01 · make_sequences(): {d["shape"][0]} secuencias de forma (batch, {b}, {f}); TR {d["splits"]["TR"]} · VA {d["splits"]["VA"]} · TE {d["splits"]["TE"]} en orden temporal; escaladores X e y ajustados en TR.',
                 '02–04 · nn.LSTM / nn.GRU(len(FEATURES), hidden, batch_first=True[, bidirectional=True]).',
                 '05 · h = state[0] si LSTM (h_n, c_n), state si GRU (h_n). 06 · summary = cat(h[-2], h[-1]) si bidireccional, si no h[-1].',
                 '07 · self.head = nn.Linear(2·hidden si bidireccional, si no hidden, 1) → retorno escalar.',
                 '08 · _, state = self.rnn(x): cada ventana empieza con memoria nueva (no se arrastran estados entre ventanas).',
                 f'09 · selected = comparison.sort_values("validation_MSE").iloc[0].modelo → «{o["selected"]}».',
                 f'10 · pred = y_scaler.inverse_transform(...): en TE, MSE {te["selected_MSE"]:.4g} frente a Ridge {te["ridge_MSE"]:.4g} ({gain:+.1%}) y {te["zero_MSE"]:.4g} de predecir 0.',
                 (f'El modelo recurrente mejora a Ridge fuera de muestra.' if gain > 0 else 'El modelo recurrente NO mejora a Ridge fuera de muestra: la complejidad extra no aporta aquí.')
                 + (' Ninguno mejora a predecir 0: no hay señal explotable en estos indicadores.' if min(te['selected_MSE'], te['ridge_MSE']) >= te['zero_MSE'] else '')]
        if o['next'] is not None:
            notes.insert(0, f'Estimación del siguiente intradía ({_x(frame.index[d["live_row"]])} → sesión siguiente): {o["next"]:+.2f} {unit} con {o["selected"]}.')
        actual = np.asarray(o['actual'])
        charts = [{'title': 'Selección por validación (MSE)', 'x_key': 'modelo', 'series': ['validation_MSE'], 'rows': table, 'chart_type': 'bar'},
                  {'title': f'TE: real vs {o["selected"]} vs Ridge ({unit})', 'x_key': 'x', 'series': ['real', o['selected'], 'Ridge'],
                   'rows': [{'x': _x(frame.index[r]), 'real': float(a), o['selected']: float(s), 'Ridge': float(g)}
                            for r, a, s, g in zip(rows_te, actual, o['prediction'], d['ridge_te'])], 'chart_type': 'line'},
                  {'title': 'Pérdida de validación por época', 'x_key': 'epoch', 'series': list(d['histories']),
                   'rows': [{'epoch': e + 1, **{k: h[e]['validation'] for k, h in d['histories'].items() if e < len(h) and 'validation' in h[e]}}
                            for e in range(max(len(h) for h in d['histories'].values()))], 'chart_type': 'line'}]
        d.setdefault('layers', [])
    else:
        score, flag = np.asarray(o['score']), np.asarray(o['is_anomaly'])
        out['AnomalyScore'], out['IsAnomaly'] = score, flag.astype(float)
        dims = np.asarray(o['dimension_error'])
        rows = [{'x': _x(i), 'score': float(s), 'umbral': o['threshold']} for i, s in zip(frame.index, score)]
        top = np.argsort(-score)[:10]
        table = [{'fecha': _x(frame.index[i]), 'score': float(score[i]), 'anómala': 'sí' if flag[i] else 'no', 'dimensión principal': cols[int(dims[i].argmax())]} for i in top]
        metrics = {**meta, 'Anomalías': d['anomalies'], 'Tras el ajuste': d['anomalies_after_fit'], 'Tasa': o['anomaly_rate'], 'Umbral': o['threshold']}
        notes = [f'Autoencoder {len(cols)}→{tool.params.latent} ajustado y con umbral fijado en las primeras {d["fit_rows"]} filas; columnas: {", ".join(cols)}.',
                 f'{d["anomalies"]} filas superan el umbral ({o["anomaly_rate"]:.1%}); {d["anomalies_after_fit"]} están después del tramo de ajuste.']
        charts = [{'title': 'Score de anomalía (error de reconstrucción)', 'x_key': 'x', 'series': ['score', 'umbral'], 'rows': rows[-2000:], 'chart_type': 'line'},
                  {'title': 'Error medio por dimensión en filas anómalas', 'x_key': 'columna', 'series': ['error'],
                   'rows': [{'columna': c, 'error': float(dims[flag].mean(0)[k]) if flag.any() else 0.0} for k, c in enumerate(cols)], 'chart_type': 'bar'}]
    training = d.get('training') or {}
    if training.get('history'):
        charts.append({'title': f'Pérdida por época · {training["optimizer"]} (entrenamiento / validación interna)', 'x_key': 'epoch',
                       'series': ['train', 'validation'] if 'validation' in training['history'][0] else ['train'], 'rows': training['history'], 'chart_type': 'line'})
        notes.append(f'Entrenamiento: {training["epochs_run"]} épocas con {training["optimizer"]}' + (f'; mejor época {training["best_epoch"]} en la validación interna ({training["val_rows"]} filas)' if training.get('val_rows') else '')
                     + ('; parada temprana.' if training.get('early_stopped') else '.'))
    if d.get('layers'):
        notes.insert(0, f'Arquitectura ({layer_source}): {architecture_line(d["layers"])} · {d.get("parameters", 0):,} parámetros.')
        metrics['Parámetros'] = d.get('parameters')
    out.attrs = {**frame.attrs}
    report = {'study': True, 'topic': kind, 'selected': cls.spec.title, 'scope': 'holdout / fuera de muestra', 'features': [],
              'splits': ({'train': d['splits']['TR'], 'validation': d['splits']['VA'], 'test': d['splits']['TE']} if isinstance(d.get('splits'), dict)
                         else {'train': d.get('fit_windows', d.get('fit_rows', 0)), 'validation': 0, 'test': d.get('holdout_windows', 0)}),
              'metrics': metrics, 'table': table, 'charts': charts,
              'notes': [*notes, f'Herramienta independiente {cls.spec.name} v{cls.spec.version} · pesos {result.weights} · {result.device} · descargada al terminar.']}
    return out, report
