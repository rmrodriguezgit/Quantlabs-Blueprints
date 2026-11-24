"""ML Pipeline: eight independent, configurable steps that pass a DataFrame from one to the next.

Select columns → Returns → Clean → Split (TR/VA/TE) → Train → Evaluate → Predict → Visualize.

Everything travels on the ordinary ``frame`` pin so each step can also feed Plot, Risk or any other
frame tool. The split is a ``Split`` column (TR / VA / TE, plus FUT for rows whose future target is
not known yet) and the metadata lives in ``frame.attrs['ml']``. Fitted estimators are kept in a small
in-process registry keyed by id (pandas deep-copies attrs, so the model itself is not stored there).

Leakage rules: every statistic that is *fitted* (imputation medians/means, outlier bounds, scalers,
the estimator, VA model selection) uses TR rows only; VA is used only to choose among candidates; TE
is scored once. Chronological splitting is the default because these are time series.
"""
from __future__ import annotations

import uuid
import warnings
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import pandas as pd

Progress = Callable[[dict[str, Any]], None] | None
SPLITS = ('TR', 'VA', 'TE')
_MODELS: 'OrderedDict[str, Any]' = OrderedDict()


def _names(value: Any) -> list[str]:
    return [s.strip() for s in str(value or '').split(',') if s.strip()]


def _attrs(frame: pd.DataFrame, **ml: Any) -> dict[str, Any]:
    base = {k: v for k, v in frame.attrs.items() if k != 'ml'}
    return {**base, 'raw_rows': frame.attrs.get('raw_rows', len(frame)), 'ml': {**frame.attrs.get('ml', {}), **ml}}


def _missing(frame: pd.DataFrame, names: list[str]) -> None:
    lost = [n for n in names if n not in frame.columns]
    if lost:
        raise ValueError(f'Columnas inexistentes: {", ".join(lost)}. Disponibles: {", ".join(map(str, frame.columns))}')


def _report(topic: str, **body: Any) -> dict[str, Any]:
    return {'study': True, 'topic': topic, 'selected': body.pop('selected', ''), 'scope': body.pop('scope', ''),
            'features': body.pop('features', []), 'splits': body.pop('splits', {'train': 0, 'validation': 0, 'test': 0}),
            'metrics': body.pop('metrics', {}), 'table': body.pop('table', []), 'charts': body.pop('charts', []),
            'notes': body.pop('notes', []), **body}


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.attrs.get('ml', {}).get('counts'):
        return frame.attrs['ml']['counts']
    s = frame['Split'] if 'Split' in frame else pd.Series(dtype=str)
    return {'train': int((s == 'TR').sum()), 'validation': int((s == 'VA').sum()), 'test': int((s == 'TE').sum())}


def _x(v: Any) -> str:
    return v.isoformat() if hasattr(v, 'isoformat') else str(v)


# 1 · Select columns -------------------------------------------------------------------------------
def select_columns(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(p.get('mode', 'keep'))
    if mode not in ('keep', 'drop'):
        raise ValueError('Modo: keep o drop.')
    names = _names(p.get('columns', ''))
    _missing(frame, names)
    always = _names(p.get('always_keep', ''))
    _missing(frame, always)
    if mode == 'keep':
        chosen = names or list(frame.columns)
    else:
        chosen = [c for c in frame.columns if c not in names]
    chosen = list(dict.fromkeys([*chosen, *[a for a in always if a not in chosen]]))
    if p.get('numeric_only', True):
        dropped_text = [c for c in chosen if not pd.api.types.is_numeric_dtype(frame[c]) and c != 'Split']
        chosen = [c for c in chosen if c not in dropped_text]
    else:
        dropped_text = []
    if not chosen:
        raise ValueError('La selección deja la tabla sin columnas.')
    out = frame[chosen].copy()
    for old, new in (pair.split('=', 1) for pair in _names(p.get('rename', '')) if '=' in pair):
        if old.strip() in out.columns:
            out = out.rename(columns={old.strip(): new.strip()})
    out.attrs = _attrs(frame)
    removed = [c for c in frame.columns if c not in chosen]
    notes = [f'{len(out.columns)} de {len(frame.columns)} columnas conservadas ({mode}).']
    if dropped_text:
        notes.append(f'Descartadas por no ser numéricas: {", ".join(map(str, dropped_text))}.')
    table = [{'columna': str(c), 'tipo': str(frame[c].dtype), 'nulos': int(frame[c].isna().sum()),
              'estado': 'conservada' if c in chosen else 'eliminada'} for c in frame.columns]
    return out, _report('ml_select', selected=', '.join(map(str, out.columns)), scope='sin ajuste (no aprende de los datos)',
                        features=list(map(str, out.columns)), table=table, notes=notes, metrics={'columnas': len(out.columns), 'eliminadas': len(removed)})


# 2 · Simple / log returns -------------------------------------------------------------------------
def returns(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    kind = str(p.get('return_type', 'log'))
    if kind not in ('simple', 'log', 'both'):
        raise ValueError('Tipo de rendimiento: simple, log o both.')
    periods = int(p.get('periods', 1))
    if not 1 <= periods <= 252:
        raise ValueError('Periodos del rendimiento: 1–252.')
    names = _names(p.get('columns', 'Close')) or ['Close']
    _missing(frame, names)
    out = frame.copy()
    created: list[str] = []
    for name in names:
        price = pd.to_numeric(frame[name], errors='coerce').astype(float)
        if kind in ('log', 'both') and (price <= 0).any():
            raise ValueError(f'{name} tiene valores ≤ 0: el rendimiento logarítmico no existe; usa simple.')
        suffix = '' if periods == 1 else f'_{periods}'
        if kind in ('simple', 'both'):
            col = f'{p.get("prefix_simple", "Ret_")}{name}{suffix}'
            out[col] = price.pct_change(periods, fill_method=None)
            created.append(col)
        if kind in ('log', 'both'):
            col = f'{p.get("prefix_log", "LogRet_")}{name}{suffix}'
            out[col] = np.log(price / price.shift(periods))
            created.append(col)
        if p.get('replace', False):
            out = out.drop(columns=[name])
    if p.get('drop_na', True):
        out = out.dropna(subset=created)
    out.attrs = _attrs(frame)
    rows = [{'x': _x(i), **{c: float(out.at[i, c]) for c in created if pd.notna(out.at[i, c])}} for i in out.index[-500:]]
    table = [{'columna': c, 'media': float(out[c].mean()), 'desv': float(out[c].std()), 'mín': float(out[c].min()), 'máx': float(out[c].max())} for c in created]
    notes = [f'Rendimiento {kind} a {periods} periodo(s): simple = Pₜ/Pₜ₋ₖ − 1, log = ln(Pₜ/Pₜ₋ₖ).',
             'El rendimiento en t usa sólo precios hasta t (sin mirar al futuro). Para predecir, Dividir crea el objetivo desplazado.']
    if not p.get('drop_na', True):
        notes.append(f'Las primeras {periods} filas quedan sin rendimiento (NaN); Limpieza puede tratarlas.')
    return out, _report('ml_returns', selected=', '.join(created), scope='sin ajuste', features=created, table=table, notes=notes,
                        charts=[{'title': 'Rendimientos', 'x_key': 'x', 'series': created, 'rows': rows, 'chart_type': 'line'}],
                        metrics={'filas': len(out), 'columnas_nuevas': len(created)})


# 3 · Cleaning -------------------------------------------------------------------------------------
def clean(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = str(p.get('missing', 'ffill'))
    outliers = str(p.get('outliers', 'none'))
    if missing not in ('drop', 'ffill', 'bfill', 'ffill_bfill', 'interpolate', 'zero', 'mean', 'median', 'none'):
        raise ValueError('Faltantes: drop, ffill, bfill, ffill_bfill, interpolate, zero, mean, median o none.')
    if outliers not in ('none', 'clip_zscore', 'clip_quantile', 'remove_zscore', 'remove_iqr'):
        raise ValueError('Atípicos: none, clip_zscore, clip_quantile, remove_zscore o remove_iqr.')
    fit_fraction = float(p.get('fit_fraction', 0.7))
    z = float(p.get('z_threshold', 4))
    q_low, q_high = float(p.get('q_low', 0.01)), float(p.get('q_high', 0.99))
    if not 0.1 <= fit_fraction <= 1 or not 1 <= z <= 20 or not 0 <= q_low < q_high <= 1:
        raise ValueError('Fracción de ajuste 0.1–1, z 1–20 y 0 ≤ q_low < q_high ≤ 1.')
    out = frame.copy()
    log: list[dict[str, Any]] = []
    before = len(out)
    names = _names(p.get('columns', '')) or [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    _missing(out, names)
    if p.get('sort_index', True) and isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
    if p.get('drop_duplicates', True):
        dup = out.index.duplicated(keep='last') if isinstance(out.index, pd.DatetimeIndex) else out.duplicated().to_numpy()
        if dup.any():
            out = out[~dup]
            log.append({'paso': 'duplicados', 'detalle': f'{int(dup.sum())} filas duplicadas eliminadas'})
    num = out[names].apply(pd.to_numeric, errors='coerce').astype(float)
    inf = int(np.isinf(num.to_numpy()).sum())
    if inf:
        num = num.replace([np.inf, -np.inf], np.nan)
        log.append({'paso': 'infinitos', 'detalle': f'{inf} valores ±inf convertidos en faltantes'})
    out[names] = num
    fit_rows = max(2, int(len(out) * fit_fraction))
    fit = out[names].iloc[:fit_rows]
    nulls = int(out[names].isna().sum().sum())
    if missing == 'drop':
        out = out.dropna(subset=names)
    elif missing in ('ffill', 'ffill_bfill'):
        out[names] = out[names].ffill()
        if missing == 'ffill_bfill':
            out[names] = out[names].bfill()
    elif missing == 'bfill':
        out[names] = out[names].bfill()
    elif missing == 'interpolate':
        out[names] = out[names].interpolate(limit_direction='forward')
    elif missing == 'zero':
        out[names] = out[names].fillna(0.0)
    elif missing in ('mean', 'median'):
        out[names] = out[names].fillna(getattr(fit, missing)())
    if nulls:
        log.append({'paso': 'faltantes', 'detalle': f'{nulls} faltantes · método {missing}' + (f' (estadístico de las primeras {fit_rows} filas)' if missing in ('mean', 'median') else '')})
    if missing in ('bfill', 'ffill_bfill', 'interpolate'):
        log.append({'paso': 'aviso', 'detalle': 'bfill/interpolate usan valores posteriores: hay fuga de información si se aplica a rasgos del futuro.'})
    clipped = removed = 0
    if outliers != 'none':
        cols = [c for c in names if c != 'Split']
        mu, sd = fit[cols].mean(), fit[cols].std().replace(0, np.nan)
        if outliers == 'clip_zscore':
            lo, hi = mu - z * sd, mu + z * sd
        elif outliers == 'clip_quantile':
            lo, hi = fit[cols].quantile(q_low), fit[cols].quantile(q_high)
        elif outliers == 'remove_iqr':
            q1, q3 = fit[cols].quantile(0.25), fit[cols].quantile(0.75)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        else:
            lo, hi = mu - z * sd, mu + z * sd
        values = out[cols]
        mask = values.lt(lo, axis=1) | values.gt(hi, axis=1)
        if outliers.startswith('clip'):
            clipped = int(mask.sum().sum())
            out[cols] = values.clip(lo, hi, axis=1)
        else:
            bad = mask.any(axis=1)
            removed = int(bad.sum())
            out = out[~bad]
        log.append({'paso': 'atípicos', 'detalle': f'{outliers}: {clipped} recortados, {removed} filas eliminadas (límites de las primeras {fit_rows} filas)'})
    if p.get('drop_constant', True):
        const = [c for c in names if c in out and out[c].nunique(dropna=True) <= 1]
        if const:
            out = out.drop(columns=const)
            log.append({'paso': 'constantes', 'detalle': f'Eliminadas: {", ".join(const)}'})
    left = int(out.select_dtypes('number').isna().sum().sum())
    if left and p.get('drop_remaining_na', True):
        out = out.dropna()
        log.append({'paso': 'faltantes restantes', 'detalle': f'{left} valores aún faltantes → filas eliminadas'})
    if len(out) < 10:
        raise ValueError(f'Tras la limpieza quedan {len(out)} filas; relaja los filtros (mínimo 10).')
    out.attrs = _attrs(frame)
    if not log:
        log.append({'paso': 'sin cambios', 'detalle': 'No había duplicados, faltantes, infinitos ni atípicos que tratar.'})
    notes = [f'{before} → {len(out)} filas. Estadísticos ajustados en las primeras {int(fit_fraction * 100)} % de filas para no mirar el futuro (alinéalo con la fracción TR de Dividir).']
    return out, _report('ml_clean', selected=f'{missing} · {outliers}', scope=f'ajuste en las primeras {fit_rows} filas',
                        features=[c for c in names if c in out], table=log, notes=notes,
                        metrics={'filas_antes': before, 'filas_después': len(out), 'faltantes': nulls, 'recortados': clipped, 'eliminadas': before - len(out)})


# 4 · Split TR / VA / TE ---------------------------------------------------------------------------
def _scaler(method: str, train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if method == 'standard':
        return train.mean(), train.std(ddof=0).replace(0, 1.0)
    if method == 'minmax':
        return train.min(), (train.max() - train.min()).replace(0, 1.0)
    if method == 'robust':
        return train.median(), (train.quantile(0.75) - train.quantile(0.25)).replace(0, 1.0)
    return pd.Series(0.0, index=train.columns), pd.Series(1.0, index=train.columns)


def split(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = str(p.get('target', '')).strip()
    if not target:
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        target = next((c for c in numeric if 'ret' in c.lower()), numeric[0] if numeric else '')
    _missing(frame, [target])
    mode, horizon = str(p.get('target_mode', 'next')), int(p.get('horizon', 1))
    task = str(p.get('task', 'auto'))
    method, scaling = str(p.get('method', 'chronological')), str(p.get('scaling', 'standard'))
    tr, va = float(p.get('train_ratio', 0.7)), float(p.get('val_ratio', 0.15))
    gap, seed = int(p.get('gap', 0)), int(p.get('seed', 42))
    if mode not in ('same', 'next', 'direction') or task not in ('auto', 'regression', 'classification'):
        raise ValueError('Objetivo: same/next/direction · tarea: auto/regression/classification.')
    if method not in ('chronological', 'random') or scaling not in ('none', 'standard', 'minmax', 'robust'):
        raise ValueError('Método chronological/random · escalado none/standard/minmax/robust.')
    if not 1 <= horizon <= 252 or not 0.3 <= tr <= 0.9 or not 0.05 <= va <= 0.4 or tr + va >= 0.95 or not 0 <= gap <= 500:
        raise ValueError('Horizonte 1–252, TR 0.3–0.9, VA 0.05–0.4, TR+VA < 0.95 y hueco 0–500.')
    data = frame.drop(columns=[c for c in ('Split', 'Target', 'Prediction', 'Proba') if c in frame]).copy()
    raw = pd.to_numeric(data[target], errors='coerce').astype(float)
    y = raw if mode == 'same' else raw.shift(-horizon)
    if mode == 'direction':
        y = (y > 0).astype(float).where(y.notna())
    if task == 'auto':
        task = 'classification' if mode == 'direction' or (y.dropna().nunique() <= 10 and np.allclose(y.dropna() % 1, 0)) else 'regression'
    features = _names(p.get('features', ''))
    if features:
        _missing(data, features)
    else:
        features = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and (c != target or mode != 'same')]
    if mode == 'same' and target in features:
        features.remove(target)
    if not features:
        raise ValueError('No quedan rasgos: elige columnas distintas del objetivo.')
    x = data[features].apply(pd.to_numeric, errors='coerce').astype(float)
    if x.isna().any().any() or np.isinf(x.to_numpy()).any():
        raise ValueError('Hay faltantes o infinitos en los rasgos: conecta Limpieza de datos antes de Dividir.')
    known = y.notna().to_numpy()
    idx = np.flatnonzero(known)
    n = len(idx)
    if n < 30:
        raise ValueError(f'Sólo {n} filas con objetivo conocido; se necesitan ≥ 30 para TR/VA/TE.')
    n_tr, n_va = int(n * tr), int(n * va)
    labels = np.array(['FUT'] * len(data), dtype=object)
    if method == 'chronological':
        order = idx
    else:
        order = np.random.default_rng(seed).permutation(idx)
    # the gap (embargo) drops rows between sets so a horizon-h target cannot straddle two sets
    parts = {'TR': order[:n_tr], 'VA': order[n_tr + gap:n_tr + n_va], 'TE': order[n_tr + n_va + gap:]}
    labels[known] = 'GAP'
    for name, rows in parts.items():
        labels[rows] = name
    if min(len(v) for v in parts.values()) < 5:
        raise ValueError('Algún conjunto queda con < 5 filas; reduce el hueco o cambia las proporciones.')
    if task == 'classification' and pd.Series(y.iloc[parts['TR']]).nunique() < 2:
        raise ValueError('TR sólo contiene una clase: no se puede entrenar un clasificador.')
    center, scale = _scaler(scaling, x.iloc[parts['TR']])
    xs = (x - center) / scale
    out = xs.copy()
    out['Target'] = y
    out['Split'] = labels
    out = out[out['Split'] != 'GAP'] if gap else out
    if not p.get('keep_future', True):
        out = out[out['Split'] != 'FUT']
    out.attrs = _attrs(frame)
    out.attrs['ml'] = {'target': target, 'target_mode': mode, 'horizon': horizon, 'task': task, 'features': features,
                       'scaling': scaling, 'method': method, 'center': center.to_dict(), 'scale': scale.to_dict()}
    counts = _counts(out)
    out.attrs['ml']['counts'] = counts
    rows = [{'x': _x(i), 'TR': float(v) if s == 'TR' else None, 'VA': float(v) if s == 'VA' else None, 'TE': float(v) if s == 'TE' else None}
            for i, v, s in zip(out.index, raw.reindex(out.index), out['Split'])]
    rows = [{k: v for k, v in r.items() if v is not None} for r in rows][-1500:]
    obj = {'same': f'{target} en t', 'next': f'{target} en t+{horizon}', 'direction': f'1 si {target} en t+{horizon} > 0'}[mode]
    notes = [f'Objetivo: {obj} · tarea {task}. Rasgos ({len(features)}): {", ".join(features[:12])}{"…" if len(features) > 12 else ""}.',
             f'División {method}: TR {counts["train"]} · VA {counts["validation"]} · TE {counts["test"]}' + (f' · hueco {gap} filas entre conjuntos' if gap else '') + '.',
             f'Escalado {scaling} ajustado sólo en TR y aplicado igual a VA/TE.',
             'VA sirve para elegir el modelo/hiperparámetros; TE se evalúa una sola vez al final.']
    if method == 'random':
        notes.append('Aviso: la división aleatoria mezcla pasado y futuro; en series temporales sobreestima el rendimiento.')
    fut = int((out['Split'] == 'FUT').sum())
    if fut:
        notes.append(f'{fut} fila(s) FUT: tienen rasgos pero su objetivo aún no se conoce; Realizar predicciones las usa para el pronóstico.')
    if mode == 'same':
        notes.append('Objetivo «same»: se explica el valor contemporáneo; no es un pronóstico.')
    table = [{'conjunto': k, 'filas': len(v), 'desde': _x(data.index[v].min()) if len(v) else '', 'hasta': _x(data.index[v].max()) if len(v) else '',
              'media objetivo': float(y.iloc[v].mean())} for k, v in parts.items()]
    return out, _report('ml_split', selected=f'{method} · {task}', scope='escalado ajustado en TR', features=features, splits=counts,
                        table=table, notes=notes, charts=[{'title': f'{target} por conjunto', 'x_key': 'x', 'series': ['TR', 'VA', 'TE'], 'rows': rows, 'chart_type': 'line'}],
                        metrics={'TR': counts['train'], 'VA': counts['validation'], 'TE': counts['test'], 'rasgos': len(features)})


# 5 · Train ----------------------------------------------------------------------------------------
REGRESSORS = ('Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Random Forest', 'Gradient Boosting', 'Histogram Boosting', 'SVR', 'KNN', 'MLP')
CLASSIFIERS = ('Logistic', 'Random Forest', 'Gradient Boosting', 'Histogram Boosting', 'SVC', 'KNN', 'MLP')


def _meta(frame: pd.DataFrame) -> dict[str, Any]:
    ml = frame.attrs.get('ml')
    if not ml or 'Split' not in frame or 'Target' not in frame:
        raise ValueError('Falta la división: conecta «Dividir TR/VA/TE» antes de este módulo.')
    return ml


def _factory(name: str, task: str, p: dict[str, Any]) -> Any:
    from sklearn import ensemble, linear_model, neighbors, neural_network, svm
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', module='sklearn.neural_network')
    seed, alpha = int(p.get('seed', 42)), float(p.get('alpha', 1.0))
    trees, depth, lr = int(p.get('estimators', 200)), int(p.get('max_depth', 6)), float(p.get('learning_rate', 0.05))
    k, hidden = int(p.get('neighbors', 10)), tuple(int(h) for h in _names(p.get('hidden_layers', '64,32')) or ['64'])
    if task == 'classification':
        table = {
            'Logistic': lambda: linear_model.LogisticRegression(C=1 / max(alpha, 1e-6), max_iter=2000),
            'Random Forest': lambda: ensemble.RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=3, random_state=seed, n_jobs=1),
            'Gradient Boosting': lambda: ensemble.GradientBoostingClassifier(n_estimators=trees, max_depth=min(depth, 8), learning_rate=lr, random_state=seed),
            'Histogram Boosting': lambda: ensemble.HistGradientBoostingClassifier(max_iter=trees, max_depth=depth, learning_rate=lr, random_state=seed),
            'SVC': lambda: svm.SVC(C=alpha, probability=True, random_state=seed),
            'KNN': lambda: neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance'),
            'MLP': lambda: neural_network.MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4 * alpha, learning_rate_init=min(lr, 0.01), max_iter=500, early_stopping=False, random_state=seed),
        }
    else:
        table = {
            'Linear': lambda: linear_model.LinearRegression(),
            'Ridge': lambda: linear_model.Ridge(alpha=alpha),
            'Lasso': lambda: linear_model.Lasso(alpha=alpha * 1e-3, max_iter=10000),
            'ElasticNet': lambda: linear_model.ElasticNet(alpha=alpha * 1e-3, l1_ratio=0.5, max_iter=10000),
            'Random Forest': lambda: ensemble.RandomForestRegressor(n_estimators=trees, max_depth=depth, min_samples_leaf=3, random_state=seed, n_jobs=1),
            'Gradient Boosting': lambda: ensemble.GradientBoostingRegressor(n_estimators=trees, max_depth=min(depth, 8), learning_rate=lr, random_state=seed),
            'Histogram Boosting': lambda: ensemble.HistGradientBoostingRegressor(max_iter=trees, max_depth=depth, learning_rate=lr, random_state=seed),
            'SVR': lambda: svm.SVR(C=alpha, epsilon=0.1),
            'KNN': lambda: neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance'),
            'MLP': lambda: neural_network.MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-4 * alpha, learning_rate_init=min(lr, 0.01), max_iter=500, random_state=seed),
        }
    if name not in table:
        raise ValueError(f'«{name}» no está disponible para {task}: {", ".join(table)}.')
    return table[name]()


def _score(task: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, float]:
    from sklearn import metrics as m
    if task == 'classification':
        out = {'Accuracy': float(m.accuracy_score(y, pred)), 'Precision': float(m.precision_score(y, pred, zero_division=0, average='binary' if len(np.unique(y)) <= 2 else 'macro')),
               'Recall': float(m.recall_score(y, pred, zero_division=0, average='binary' if len(np.unique(y)) <= 2 else 'macro')),
               'F1': float(m.f1_score(y, pred, zero_division=0, average='binary' if len(np.unique(y)) <= 2 else 'macro'))}
        if proba is not None and len(np.unique(y)) == 2:
            out['AUC'] = float(m.roc_auc_score(y, proba))
            out['LogLoss'] = float(m.log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]))
        return out
    mse = float(m.mean_squared_error(y, pred))
    out = {'MSE': mse, 'RMSE': float(np.sqrt(mse)), 'MAE': float(m.mean_absolute_error(y, pred)), 'R2': float(m.r2_score(y, pred)) if len(y) > 1 else float('nan')}
    nz = y != 0
    if nz.any():
        out['Acierto dirección'] = float(np.mean(np.sign(pred[nz]) == np.sign(y[nz])))
    return out


def train(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    ml = _meta(frame)
    task, features = ml['task'], ml['features']
    algorithm = str(p.get('algorithm_cls' if task == 'classification' else 'algorithm', 'Compare'))
    choices = CLASSIFIERS if task == 'classification' else REGRESSORS
    if algorithm != 'Compare' and algorithm not in choices:
        raise ValueError(f'Algoritmo para {task}: Compare, {", ".join(choices)}.')
    if not 10 <= int(p.get('estimators', 200)) <= 1000 or not 1 <= int(p.get('max_depth', 6)) <= 32 or not 1 <= int(p.get('neighbors', 10)) <= 200:
        raise ValueError('Árboles 10–1000, profundidad 1–32 y vecinos 1–200.')
    s = frame['Split']
    X = {k: frame.loc[s == k, features].to_numpy(float) for k in SPLITS}
    Y = {k: frame.loc[s == k, 'Target'].to_numpy(float) for k in SPLITS}
    if len(X['TR']) > 50000:
        raise ValueError('Máximo 50 000 filas de entrenamiento.')
    metric = 'LogLoss' if task == 'classification' else 'RMSE'
    candidates = choices if algorithm == 'Compare' else (algorithm,)
    if algorithm == 'Compare' and len(X['TR']) > 5000:
        candidates = tuple(c for c in candidates if c not in ('SVR', 'SVC', 'MLP'))
    rows, best, winner, chosen = [], float('inf'), None, ''
    for i, name in enumerate(candidates):
        model = _factory(name, task, p)
        model.fit(X['TR'], Y['TR'].astype(int) if task == 'classification' else Y['TR'])
        proba = model.predict_proba(X['VA'])[:, 1] if task == 'classification' and hasattr(model, 'predict_proba') and len(model.classes_) == 2 else None
        pred = (proba >= 0.5).astype(float) if proba is not None else model.predict(X['VA'])
        sc = _score(task, Y['VA'], pred, proba)
        key = metric if metric in sc else ('Accuracy' if task == 'classification' else 'RMSE')
        value = -sc[key] if key == 'Accuracy' else sc[key]
        rows.append({'algoritmo': name, **{f'VA {k}': v for k, v in sc.items()}})
        if value < best:
            best, winner, chosen = value, model, name
        if progress:
            progress({'detail': f'{name}: VA {key} {sc[key]:.4g} · {i + 1}/{len(candidates)}'})
    assert winner is not None
    if p.get('refit_train_val', False):
        winner = _factory(chosen, task, p)
        xy = np.vstack([X['TR'], X['VA']]), np.concatenate([Y['TR'], Y['VA']])
        winner.fit(xy[0], xy[1].astype(int) if task == 'classification' else xy[1])
    key = uuid.uuid4().hex
    _MODELS[key] = winner
    while len(_MODELS) > 32:
        _MODELS.popitem(last=False)
    out = frame.copy()
    allx = out[features].to_numpy(float)
    out['Prediction'] = winner.predict(allx).astype(float)
    if task == 'classification' and hasattr(winner, 'predict_proba') and len(winner.classes_) == 2:
        out['Proba'] = winner.predict_proba(allx)[:, 1]
    importance = _importance(winner, features, X['VA'], Y['VA'], task)
    out.attrs = _attrs(frame, model=key, algorithm=chosen, selection=rows, importance=importance, refit=bool(p.get('refit_train_val', False)))
    if 'Proba' in out:  # same rule as Evaluate (P >= 0.5) so TR/VA/TE figures agree
        tr_scores = _score(task, Y['TR'], (out.loc[s == 'TR', 'Proba'].to_numpy() >= 0.5).astype(float))
    else:
        tr_scores = _score(task, Y['TR'], winner.predict(X['TR']))
    va_metric = [r for r in rows if r['algoritmo'] == chosen][0]
    notes = [f'{len(candidates)} candidato(s) entrenados sólo en TR ({len(X["TR"])} filas); VA ({len(X["VA"])}) elige «{chosen}» por {metric}.',
             'TE no se ha usado todavía: conecta «Evaluar el modelo» para medirlo una sola vez.']
    if p.get('refit_train_val', False):
        notes.append('Tras elegir, el ganador se reentrenó con TR+VA; las métricas VA de la tabla son previas al reentreno.')
    gap_note = _overfit(task, tr_scores, va_metric)
    if gap_note:
        notes.append(gap_note)
    chart_metric = f'VA {metric}' if f'VA {metric}' in rows[0] else next(k for k in rows[0] if k != 'algoritmo')
    charts = [{'title': f'Selección de modelo · {chart_metric}', 'x_key': 'algoritmo', 'series': [chart_metric], 'rows': rows, 'chart_type': 'bar'}]
    if importance:
        charts.append({'title': 'Importancia de rasgos (permutación en VA)', 'x_key': 'rasgo', 'series': ['importancia'], 'rows': importance[:20], 'chart_type': 'bar'})
    return out, _report('ml_train', selected=chosen, scope='ajuste en TR · selección en VA', features=features, splits=_counts(frame),
                        table=rows, notes=notes, charts=charts, metrics={**{f'TR {k}': v for k, v in tr_scores.items()}, **{k: v for k, v in va_metric.items() if k != 'algoritmo'}})


def _overfit(task: str, tr: dict[str, float], va: dict[str, Any]) -> str:
    if task == 'classification':
        a, b = tr.get('Accuracy'), va.get('VA Accuracy')
        if a is not None and b is not None and a - b > 0.15:
            return f'Posible sobreajuste: exactitud TR {a:.1%} frente a VA {b:.1%}. Reduce profundidad/árboles o añade regularización.'
    else:
        a, b = tr.get('RMSE'), va.get('VA RMSE')
        if a and b and b > 1.5 * a:
            return f'Posible sobreajuste: RMSE VA {b:.4g} es {b / a:.1f}× el de TR {a:.4g}.'
    return ''


def _importance(model: Any, features: list[str], x: np.ndarray, y: np.ndarray, task: str) -> list[dict[str, Any]]:
    try:
        from sklearn.inspection import permutation_importance
        if len(x) < 5:
            return []
        r = permutation_importance(model, x, y.astype(int) if task == 'classification' else y, n_repeats=5, random_state=0,
                                   scoring='accuracy' if task == 'classification' else 'neg_mean_squared_error')
        rows = [{'rasgo': f, 'importancia': float(v)} for f, v in zip(features, r.importances_mean)]
        return sorted(rows, key=lambda r: -r['importancia'])
    except Exception:  # importance is a nice-to-have; never fail training for it
        return []


def _model(frame: pd.DataFrame) -> Any:
    ml = _meta(frame)
    model = _MODELS.get(ml.get('model', ''))
    if model is None:
        raise ValueError('Falta el modelo entrenado: conecta «Entrenar modelo ML» antes de este módulo (y vuelve a ejecutar).')
    return model


# 6 · Evaluate -------------------------------------------------------------------------------------
def evaluate(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    ml = _meta(frame)
    _model(frame)
    task, s = ml['task'], frame['Split']
    which = [k for k in SPLITS if p.get(f'eval_{k.lower()}', True)] or ['TE']
    threshold = float(p.get('threshold', 0.5))
    if not 0.05 <= threshold <= 0.95:
        raise ValueError('Umbral de clasificación 0.05–0.95.')
    rows, metrics = [], {}
    for k in which:
        part = frame[s == k]
        if part.empty:
            continue
        y = part['Target'].to_numpy(float)
        proba = part['Proba'].to_numpy(float) if 'Proba' in part else None
        pred = (proba >= threshold).astype(float) if proba is not None else part['Prediction'].to_numpy(float)
        sc = _score(task, y, pred, proba)
        base = _baseline(task, frame, y, k)
        rows.append({'conjunto': k, 'filas': len(part), **sc, **{f'base {n}': v for n, v in base.items()}})
        if k == 'TE' or len(which) == 1:
            metrics.update(sc)
    notes = []
    te = next((r for r in rows if r['conjunto'] == 'TE'), None)
    if te:
        if task == 'classification':
            edge = te['Accuracy'] - te['base Accuracy']
            notes.append(f'TE: exactitud {te["Accuracy"]:.1%} frente a {te["base Accuracy"]:.1%} de la clase mayoritaria de TR ({edge:+.1%}).')
            notes.append('El modelo aporta valor sólo si supera a la línea base fuera de muestra.' if edge > 0 else 'El modelo no supera a la línea base en TE: no aporta señal útil.')
        else:
            edge = 1 - te['RMSE'] / te['base RMSE'] if te['base RMSE'] else float('nan')
            notes.append(f'TE: RMSE {te["RMSE"]:.4g} frente a {te["base RMSE"]:.4g} de predecir la media de TR ({edge:+.1%} de mejora).')
            notes.append('R² < 0 en TE significa que el modelo predice peor que la media.' if te['R2'] < 0 else f'R² en TE = {te["R2"]:.3f}.')
        tr = next((r for r in rows if r['conjunto'] == 'TR'), None)
        if tr:
            k1 = 'Accuracy' if task == 'classification' else 'RMSE'
            notes.append(f'Comparación TR vs TE ({k1}): {tr[k1]:.4g} → {te[k1]:.4g}. Una gran diferencia indica sobreajuste.')
    notes.append('TE no influyó en el entrenamiento ni en la selección; es la estimación honesta del rendimiento.')
    charts, table2 = [], []
    part = frame[s == 'TE'] if 'TE' in which else frame[s == which[-1]]
    if task == 'classification' and len(part):
        y = part['Target'].astype(int).to_numpy()
        proba = part['Proba'].to_numpy(float) if 'Proba' in part else None
        pred = (proba >= threshold).astype(int) if proba is not None else part['Prediction'].astype(int).to_numpy()
        labels = sorted(set(y) | set(pred))
        table2 = [{'real \\ predicho': str(a), **{str(b): int(((y == a) & (pred == b)).sum()) for b in labels}} for a in labels]
        metrics['Matriz de confusión'] = None
        if proba is not None:
            bins = np.linspace(0, 1, 11)
            cal = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
                if mask.any():
                    cal.append({'prob_media': float(proba[mask].mean()), 'frecuencia_real': float(y[mask].mean()), 'ideal': float(proba[mask].mean())})
            charts.append({'title': 'Calibración (TE)', 'x_key': 'prob_media', 'series': ['frecuencia_real', 'ideal'], 'rows': cal, 'chart_type': 'scatter_line'})
    elif len(part):
        resid = (part['Target'] - part['Prediction']).to_numpy(float)
        charts.append({'title': 'Residuos (real − predicho)', 'x_key': 'residuo', 'series': ['residuo'], 'rows': [{'residuo': float(v)} for v in resid], 'chart_type': 'histogram'})
    charts.insert(0, {'title': 'Métrica por conjunto', 'x_key': 'conjunto', 'series': ['Accuracy' if task == 'classification' else 'RMSE', 'base Accuracy' if task == 'classification' else 'base RMSE'], 'rows': rows, 'chart_type': 'bar'})
    metrics.pop('Matriz de confusión', None)
    out = frame.copy()
    out.attrs = _attrs(frame, evaluation=rows, threshold=threshold)
    return out, _report('ml_evaluate', selected=ml.get('algorithm', ''), scope='holdout / fuera de muestra (TE)', features=ml['features'],
                        splits=_counts(frame), metrics=metrics, table=rows + ([{'—': 'Matriz de confusión (TE)'}] + table2 if table2 else []), notes=notes, charts=charts)


def _baseline(task: str, frame: pd.DataFrame, y: np.ndarray, split_name: str) -> dict[str, float]:
    train_y = frame.loc[frame['Split'] == 'TR', 'Target'].to_numpy(float)
    if task == 'classification':
        major = float(pd.Series(train_y).mode().iloc[0])
        return {'Accuracy': float(np.mean(y == major))}
    pred = np.full_like(y, train_y.mean())
    return {'RMSE': float(np.sqrt(np.mean((y - pred) ** 2)))}


# 7 · Predict --------------------------------------------------------------------------------------
def predict(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    ml = _meta(frame)
    model = _model(frame)
    scope = str(p.get('scope', 'test_future'))
    if scope not in ('test', 'future', 'test_future', 'validation', 'all'):
        raise ValueError('Alcance: test, future, test_future, validation o all.')
    wanted = {'test': ['TE'], 'future': ['FUT'], 'test_future': ['TE', 'FUT'], 'validation': ['VA'], 'all': ['TR', 'VA', 'TE', 'FUT']}[scope]
    part = frame[frame['Split'].isin(wanted)]
    if part.empty:
        raise ValueError(f'No hay filas para «{scope}». Las filas FUT aparecen si Dividir usa objetivo next/direction y conserva el futuro.')
    task, threshold = ml['task'], float(p.get('threshold', ml.get('threshold', 0.5)))
    x = part[ml['features']].to_numpy(float)
    name = str(p.get('output_column', 'Prediction')).strip() or 'Prediction'
    out = pd.DataFrame(index=part.index)
    out['Split'] = part['Split']
    out['Actual'] = part['Target']
    if task == 'classification' and hasattr(model, 'predict_proba') and len(model.classes_) == 2:
        proba = model.predict_proba(x)[:, 1]
        out['Proba'] = proba
        out[name] = (proba >= threshold).astype(float)
    else:
        out[name] = model.predict(x).astype(float)
    if task == 'regression' and p.get('interval', True):
        tr = frame[frame['Split'] == 'TR']
        resid = tr['Target'] - model.predict(tr[ml['features']].to_numpy(float))
        q = float(np.quantile(np.abs(resid), float(p.get('coverage', 0.9))))
        out['Lower'], out['Upper'] = out[name] - q, out[name] + q
    out.attrs = _attrs(frame, prediction_column=name)
    known = out['Actual'].notna()
    if p.get('signal', True):
        out['Signal'] = np.where(out[name] > (threshold if task == 'classification' else 0), 1, -1)
    fut = out[out['Split'] == 'FUT']
    notes = [f'{len(out)} predicción(es) con «{ml.get("algorithm", "")}» sobre {"/".join(wanted)}.']
    if task == 'classification':
        notes.append(f'Clase 1 si P ≥ {threshold:.2f}.')
    if 'Lower' in out:
        notes.append(f'Intervalo ±{q:.4g}: cuantil {float(p.get("coverage", 0.9)):.0%} del error absoluto en TR (aproximado; suele quedarse corto fuera de muestra).')
    if len(fut):
        last = fut.iloc[-1]
        lab = ml['target'] + (f' t+{ml["horizon"]}' if ml['target_mode'] != 'same' else '')
        val = f'P(sube) = {last["Proba"]:.1%}' if 'Proba' in fut else f'{last[name]:.4g}'
        notes.insert(0, f'Pronóstico más reciente ({_x(fut.index[-1])}): {lab} → {val}.')
    if known.any():
        hit = (np.sign(out.loc[known, name]) == np.sign(out.loc[known, 'Actual'])).mean() if task == 'regression' else (out.loc[known, name] == out.loc[known, 'Actual']).mean()
        notes.append(f'Acierto ({"dirección" if task == "regression" else "clase"}) en las filas con valor real: {hit:.1%}.')
    table = [{'fecha': _x(i), 'conjunto': r['Split'], 'real': None if pd.isna(r['Actual']) else float(r['Actual']), 'predicción': float(r[name]),
              **({'P(1)': float(r['Proba'])} if 'Proba' in out else {})} for i, r in out.tail(int(p.get('rows', 20))).iterrows()]
    metrics = {'predicciones': len(out), 'futuras': len(fut)}
    if len(fut):
        metrics['último pronóstico'] = float(fut[name].iloc[-1]) if 'Proba' not in fut else float(fut['Proba'].iloc[-1])
    return out, _report('ml_predict', selected=scope, scope='fuera de muestra' if 'TR' not in wanted else 'incluye TR (dentro de muestra)',
                        features=ml['features'], splits=_counts(frame), table=table, notes=notes, metrics=metrics)


# 8 · Visualize ------------------------------------------------------------------------------------
def visualize(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    ml = frame.attrs.get('ml') or {}
    pred_col = ml.get('prediction_column', 'Prediction')
    actual_col = 'Actual' if 'Actual' in frame else 'Target'
    if pred_col not in frame or actual_col not in frame:
        raise ValueError('Visualizar necesita predicciones: conecta Entrenar, Evaluar o Realizar predicciones.')
    splits = [k for k in ('TR', 'VA', 'TE', 'FUT') if p.get(f'show_{k.lower()}', k != 'TR')]
    view = frame[frame['Split'].isin(splits)] if 'Split' in frame and splits else frame
    if view.empty:
        raise ValueError('No hay filas en los conjuntos elegidos.')
    task = ml.get('task', 'regression')
    cap = int(p.get('max_points', 1500))
    v = view.tail(cap)
    charts: list[dict[str, Any]] = []
    value = 'Proba' if task == 'classification' and 'Proba' in v else pred_col
    if p.get('series', True):
        actual = v[actual_col]
        if value == 'Proba':  # a 0/1 series is unreadable: compare P(1) with the rolling share of 1s
            actual = actual.rolling(20, min_periods=5).mean()
        rows = [{'x': _x(i), 'real': None if pd.isna(a) else float(a), 'predicho': float(r[value]),
                 **({'inferior': float(r['Lower']), 'superior': float(r['Upper'])} if 'Lower' in v else {})} for (i, r), a in zip(v.iterrows(), actual)]
        rows = [{k: x for k, x in r.items() if x is not None} for r in rows]
        charts.append({'title': (f'P(1) vs frecuencia real de 1 (media móvil 20) · {"/".join(splits)}' if value == 'Proba' else f'Real vs predicho · {"/".join(splits)}'), 'x_key': 'x',
                       'series': ['real', 'predicho', *(['inferior', 'superior'] if 'Lower' in v else [])], 'rows': rows, 'chart_type': 'line'})
    known = v[v[actual_col].notna()]
    if p.get('scatter', True) and task == 'regression' and len(known):
        charts.append({'title': 'Dispersión real vs predicho', 'x_key': 'real', 'series': ['predicho'],
                       'rows': [{'real': float(a), 'predicho': float(b)} for a, b in zip(known[actual_col], known[pred_col])], 'chart_type': 'scatter'})
    if p.get('residuals', True) and task == 'regression' and len(known):
        charts.append({'title': 'Distribución de residuos', 'x_key': 'residuo', 'series': ['residuo'],
                       'rows': [{'residuo': float(a - b)} for a, b in zip(known[actual_col], known[pred_col])], 'chart_type': 'histogram'})
    if p.get('cumulative', True) and len(known) and ml.get('target_mode') in ('next', 'direction'):
        sig = np.where(known[pred_col] > (0.5 if task == 'classification' else 0), 1.0, -1.0 if p.get('allow_short', True) else 0.0)
        if task == 'regression':
            r = known[actual_col].to_numpy(float)
            eq = np.cumsum(sig * r)
            charts.append({'title': 'Estrategia acumulada (signo de la predicción × objetivo real)', 'x_key': 'x', 'series': ['estrategia', 'comprar y mantener'],
                           'rows': [{'x': _x(i), 'estrategia': float(a), 'comprar y mantener': float(b)} for i, a, b in zip(known.index, eq, np.cumsum(r))], 'chart_type': 'line'})
    if p.get('importance', True) and ml.get('importance'):
        charts.append({'title': 'Importancia de rasgos', 'x_key': 'rasgo', 'series': ['importancia'], 'rows': ml['importance'][:20], 'chart_type': 'bar'})
    if p.get('selection', True) and ml.get('selection') and len(ml['selection']) > 1:
        key = next((k for k in ml['selection'][0] if k.endswith(('RMSE', 'LogLoss'))), next(k for k in ml['selection'][0] if k != 'algoritmo'))
        charts.append({'title': f'Comparación de modelos ({key})', 'x_key': 'algoritmo', 'series': [key], 'rows': ml['selection'], 'chart_type': 'bar'})
    if not charts:
        raise ValueError('Activa al menos un gráfico.')
    notes = [f'{len(charts)} gráfico(s) sobre {len(v)} filas de {"/".join(splits) or "todas"}.']
    if any(c['title'].startswith('Estrategia') for c in charts):
        notes.append('La curva de estrategia es ilustrativa: suma de rendimientos sin costes ni deslizamiento; usa Backtest/Risk para un análisis serio.')
    if ml.get('evaluation'):
        notes.append('Métricas por conjunto: ' + '; '.join(f'{r["conjunto"]} ' + ', '.join(f'{k} {r[k]:.4g}' for k in ('RMSE', 'R2', 'Accuracy', 'AUC') if k in r) for r in ml['evaluation']))
    out = frame.copy()
    out.attrs = _attrs(frame)
    return out, _report('ml_visualize', selected=ml.get('algorithm', ''), scope='/'.join(splits), features=ml.get('features', []),
                        splits=_counts(frame) if 'Split' in frame else {'train': 0, 'validation': 0, 'test': 0},
                        charts=charts, notes=notes, metrics={k: v for k, v in (ml.get('evaluation') or [{}])[-1].items() if isinstance(v, float)})


STEPS = {'ml_select': select_columns, 'ml_returns': returns, 'ml_clean': clean, 'ml_split': split,
         'ml_train': train, 'ml_evaluate': evaluate, 'ml_predict': predict, 'ml_visualize': visualize}
