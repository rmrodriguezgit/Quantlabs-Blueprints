"""Autoencoders (Unidad 8): diez herramientas independientes que se pasan un DataFrame.

Validar/dividir → Estandarizar → PCA → Encoder → Decoder → Ruido → Entrenar AE/DAE →
Seleccionar por reconstrucción → Ridge sobre factores → Medir en test.

Todo viaja por el pin ``frame``. La columna ``Split`` (TR/VA/TE) marca los conjuntos y la
configuración vive en ``frame.attrs['ae']``. Los modelos ajustados (PCA, AE, DAE, Ridge) se
guardan en un registro en proceso por id (pandas copia attrs en profundidad).

Reglas sin fuga: escalador, PCA, pesos y Ridge se ajustan sólo con TR; VA elige el modelo y
detiene el entrenamiento; TE se mide una vez al final.
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from . import ae_core as core

Progress = Callable[[dict[str, Any]], None] | None
DEFAULT_FEATURES = 'LogReturn,Volatility,RSI,EMA,SMA,VWAP,MACD,MACDSignal,BBUpper,BBMiddle,BBLower'
ACTS = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'GELU': nn.GELU, 'SiLU': nn.SiLU, 'LeakyReLU': nn.LeakyReLU}
_MODELS: 'OrderedDict[str, Any]' = OrderedDict()


# ── utilidades ─────────────────────────────────────────────────────────────────
def _keep(obj: Any) -> str:
    key = uuid.uuid4().hex[:12]
    _MODELS[key] = obj
    while len(_MODELS) > 64:
        _MODELS.popitem(last=False)
    return key


def _get(key: str | None) -> Any:
    if not key or key not in _MODELS:
        raise ValueError('El modelo ya no está en memoria: vuelve a ejecutar el Blueprint completo.')
    return _MODELS[key]


def _names(value: Any) -> list[str]:
    return [s.strip() for s in str(value or '').split(',') if s.strip()]


def _x(v: Any) -> str:
    return v.isoformat() if hasattr(v, 'isoformat') else str(v)


def _ae(frame: pd.DataFrame, need: str = 'Split', step: str = '1 · Validar y dividir') -> dict[str, Any]:
    ae = frame.attrs.get('ae') or {}
    if need not in ae and need not in frame:
        raise ValueError(f'Falta «{step}» antes de este nodo.')
    return ae


def _out(frame: pd.DataFrame, data: pd.DataFrame | None = None, **ae: Any) -> pd.DataFrame:
    out = (frame if data is None else data).copy()
    out.attrs = {**{k: v for k, v in frame.attrs.items() if k != 'ae'}, 'raw_rows': frame.attrs.get('raw_rows', len(frame)),
                 'ae': {**frame.attrs.get('ae', {}), **ae}}
    return out


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    s = frame['Split'] if 'Split' in frame else pd.Series(dtype=str)
    return {'train': int((s == 'TR').sum()), 'validation': int((s == 'VA').sum()), 'test': int((s == 'TE').sum())}


def _report(topic: str, frame: pd.DataFrame, **body: Any) -> dict[str, Any]:
    ae = frame.attrs.get('ae', {})
    return {'study': True, 'topic': topic, 'selected': body.pop('selected', ''), 'scope': body.pop('scope', ''),
            'features': body.pop('features', ae.get('features', [])), 'splits': body.pop('splits', _counts(frame)),
            'metrics': body.pop('metrics', {}), 'table': body.pop('table', []), 'charts': body.pop('charts', []),
            'notes': body.pop('notes', []), **body}


def _block(frame: pd.DataFrame, split: str) -> np.ndarray:
    ae = _ae(frame)
    x = frame.loc[frame['Split'] == split, ae['features']].to_numpy(np.float32)
    if not len(x):
        raise ValueError(f'El conjunto {split} está vacío.')
    return x


def _y(frame: pd.DataFrame, split: str) -> np.ndarray:
    return frame.loc[frame['Split'] == split, 'y_bps'].to_numpy(float)


def _compressors(frame: pd.DataFrame) -> dict[str, core.Compressor]:
    ae = frame.attrs.get('ae', {})
    return {name: core.Compressor(name, _get(key)) for name, key in ae.get('models', {}).items()}


# ── 1 · Validar y dividir (controles de entrada + división temporal) ──────────
def split(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    features = _names(p.get('features')) or _names(DEFAULT_FEATURES)
    target, horizon = str(p.get('target', 'LogReturn')).strip(), int(p.get('horizon', 16))
    tr, va, gap = float(p.get('train_ratio', .65)), float(p.get('val_ratio', .15)), int(p.get('gap', 0))
    if len(set(features)) != len(features):
        raise ValueError('Hay indicadores repetidos.')
    lost = [c for c in [*features, target] if c not in frame.columns]
    if lost:
        raise ValueError(f'Columnas inexistentes: {", ".join(lost)}. Disponibles: {", ".join(map(str, frame.columns))}')
    if not 1 <= horizon <= 252 or not 0.3 <= tr <= 0.9 or not 0.05 <= va <= 0.4 or tr + va >= 0.95 or not 0 <= gap <= 500:
        raise ValueError('Horizonte 1–252, TR 0.3–0.9, VA 0.05–0.4, TR+VA < 0.95 y hueco 0–500.')
    data = frame[features].apply(pd.to_numeric, errors='coerce').astype(float)
    r = pd.to_numeric(frame[target], errors='coerce').astype(float)
    # objetivo: rendimiento acumulado de las próximas h filas, en puntos básicos (sin mirar el presente)
    data['y_bps'] = 1e4 * r[::-1].rolling(horizon).sum()[::-1].shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    core.validate_frame(data, features, 'y_bps')
    n = len(data)
    if n < 60:
        raise ValueError(f'Sólo {n} filas válidas; se necesitan ≥ 60 para TR/VA/TE.')
    n_tr, n_va = int(n * tr), int(n * va)
    # embargo: el objetivo de TR mira h filas hacia delante → al menos h filas de hueco
    emb = max(gap, horizon)
    labels = np.array(['GAP'] * n, dtype=object)
    labels[:n_tr] = 'TR'
    labels[n_tr + emb:n_tr + n_va] = 'VA'
    labels[n_tr + n_va + emb:] = 'TE'
    data['Split'] = labels
    data = data[data['Split'] != 'GAP']
    counts = _counts(data)
    if min(counts.values()) < 8:
        raise ValueError('Algún conjunto queda con < 8 filas; usa más historia o reduce horizonte/hueco.')
    out = _out(frame, data, features=features, target=target, horizon=horizon, models={}, arch={})
    rows = [{'x': _x(i), s: float(v)} for i, v, s in zip(out.index, out['y_bps'], out['Split'])][-1500:]
    table = [{'conjunto': k, 'filas': int((out['Split'] == k).sum()), 'desde': _x(out.index[out['Split'] == k].min()),
              'hasta': _x(out.index[out['Split'] == k].max()), 'media y_bps': float(out.loc[out['Split'] == k, 'y_bps'].mean())}
             for k in ('TR', 'VA', 'TE')]
    return out, _report('ae_split', out, selected=f'{len(features)} indicadores · y_bps a {horizon} filas', scope='cronológico',
                        table=table, metrics={'TR': counts['train'], 'VA': counts['validation'], 'TE': counts['test'], 'indicadores': len(features)},
                        charts=[{'title': 'y_bps por conjunto', 'x_key': 'x', 'series': ['TR', 'VA', 'TE'], 'rows': rows, 'chart_type': 'line'},
                                {'title': 'Distribución de y_bps (TR)', 'x_key': 'y_bps', 'series': ['y_bps'], 'chart_type': 'histogram',
                                 'rows': [{'y_bps': float(v)} for v in out.loc[out['Split'] == 'TR', 'y_bps']]}],
                        notes=[f'Controles superados: columnas presentes, valores finitos, sin columnas constantes, fechas ordenadas.',
                               f'Embargo de {emb} filas entre conjuntos para que el objetivo a {horizon} filas no cruce de TR a VA/TE.'])


# ── 2 · Estandarizar con entrenamiento (segmento 01) ──────────────────────────
def standardize(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _ae(frame)
    f = ae['features']
    tr = frame.loc[frame['Split'] == 'TR', f]
    mean, std = tr.mean(), tr.std(ddof=0)
    if (std == 0).any():
        raise ValueError(f'Columnas constantes en TR: {", ".join(std.index[std == 0])}')
    data = frame.copy()
    data[f] = ((frame[f] - mean) / std).astype(np.float32)
    out = _out(frame, data, scaled=True, scaler={'mean': mean.to_dict(), 'std': std.to_dict()})
    table = [{'indicador': c, 'media TR': float(mean[c]), 'desv. TR': float(std[c]),
              'media VA (escalada)': float(data.loc[data['Split'] == 'VA', c].mean()),
              'media TE (escalada)': float(data.loc[data['Split'] == 'TE', c].mean())} for c in f]
    return out, _report('ae_standardize', out, selected='StandardScaler · fit TR', scope='TR', table=table,
                        charts=[{'title': 'Desviación de cada indicador en TR (antes de escalar)', 'x_key': 'indicador', 'series': ['desv. TR'],
                                 'rows': [{'indicador': c, 'desv. TR': float(std[c])} for c in f], 'chart_type': 'bar'}],
                        metrics={'max |media TR escalada|': float(data.loc[data['Split'] == 'TR', f].mean().abs().max())},
                        notes=['El escalador se ajusta sólo con TR; VA/TE se transforman con la misma media y desviación (sus medias no son 0: es lo esperado).'])


def _need_scaled(frame: pd.DataFrame) -> dict[str, Any]:
    ae = _ae(frame)
    if not ae.get('scaled'):
        raise ValueError('Conecta «2 · Estandarizar (TR)» antes de este nodo.')
    return ae


# ── 3 · Referencia PCA (segmento 02) ──────────────────────────────────────────
def pca(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _need_scaled(frame)
    k = int(p.get('components', 3))
    if not 1 <= k < len(ae['features']):
        raise ValueError(f'Componentes entre 1 y {len(ae["features"]) - 1}.')
    model = PCA(n_components=k).fit(_block(frame, 'TR'))
    data = frame.copy()
    comps = model.transform(frame[ae['features']].to_numpy(np.float32))
    for i in range(k):
        data[f'PC{i + 1}'] = comps[:, i]
    out = _out(frame, data, models={**ae.get('models', {}), 'PCA': _keep(model)}, latent=k)
    ev = model.explained_variance_ratio_
    va = _block(frame, 'VA')
    mse = core.reconstruction_mse(model.inverse_transform(model.transform(va)), va)
    return out, _report('ae_pca', out, selected=f'PCA · {k} componentes', scope='fit TR',
                        metrics={'varianza explicada': float(ev.sum()), 'MSE reconstrucción VA': mse},
                        table=[{'componente': f'PC{i + 1}', 'varianza explicada': float(v), 'acumulada': float(ev[:i + 1].sum())} for i, v in enumerate(ev)],
                        charts=[{'title': 'Varianza explicada por componente', 'x_key': 'componente', 'series': ['varianza explicada'],
                                 'rows': [{'componente': f'PC{i + 1}', 'varianza explicada': float(v)} for i, v in enumerate(ev)], 'chart_type': 'bar'}],
                        notes=['Referencia lineal: el autoencoder debe mejorar su MSE de reconstrucción en VA para justificar la no linealidad.'])


# ── 4 · Encoder y 5 · Decoder (segmentos 03–05) ───────────────────────────────
def _arch_report(out: pd.DataFrame, topic: str, layers: list[str], note: str):
    return _report(topic, out, selected=' → '.join(layers), scope='arquitectura',
                   table=[{'capa': i + 1, 'definición': l} for i, l in enumerate(layers)], notes=[note])


def encoder(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _need_scaled(frame)
    h, k, act = int(p.get('hidden', 32)), int(p.get('latent', 3)), str(p.get('activation', 'ReLU'))
    d = len(ae['features'])
    if act not in ACTS or not 2 <= h <= 1024 or not 1 <= k < d:
        raise ValueError(f'Ocultas 2–1024, latente 1–{d - 1} y activación {"/".join(ACTS)}.')
    out = _out(frame, arch={'encoder': {'hidden': h, 'latent': k, 'activation': act}})
    params = d * h + h + h * k + k
    r = _arch_report(out, 'ae_encoder', [f'Lineal({d}, {h})', act, f'Lineal({h}, {k})'],
                     f'Comprime {d} indicadores a un código latente de {k} números ({params} parámetros).')
    r['metrics'] = {'parámetros': params, 'latente': k}
    return out, r


def decoder(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _need_scaled(frame)
    enc = ae.get('arch', {}).get('encoder')
    if not enc:
        raise ValueError('Conecta «4 · Encoder» antes del Decoder.')
    h, act, d, k = int(p.get('hidden', 32)), str(p.get('activation', 'ReLU')), len(ae['features']), enc['latent']
    if act not in ACTS or not 2 <= h <= 1024:
        raise ValueError(f'Ocultas 2–1024 y activación {"/".join(ACTS)}.')
    out = _out(frame, arch={**ae['arch'], 'decoder': {'hidden': h, 'activation': act}})
    model = build(out)
    total = sum(t.numel() for t in model.parameters())
    r = _arch_report(out, 'ae_decoder', [f'Lineal({k}, {h})', act, f'Lineal({h}, {d})'],
                     f'Reconstrucción x ({d}) → z ({k}) → x̂ ({d}); forward = decoder(encoder(x)). Total {total} parámetros.')
    r['metrics'] = {'parámetros totales': total}
    return out, r


class FlexAE(nn.Module):
    def __init__(self, d: int, enc: dict[str, Any], dec: dict[str, Any]):
        super().__init__()
        k = enc['latent']
        self.encoder = nn.Sequential(nn.Linear(d, enc['hidden']), ACTS[enc['activation']](), nn.Linear(enc['hidden'], k))
        self.decoder = nn.Sequential(nn.Linear(k, dec['hidden']), ACTS[dec['activation']](), nn.Linear(dec['hidden'], d))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build(frame: pd.DataFrame) -> FlexAE:
    ae = frame.attrs.get('ae', {})
    arch = ae.get('arch', {})
    if 'encoder' not in arch or 'decoder' not in arch:
        raise ValueError('Conecta «4 · Encoder» y «5 · Decoder» antes de entrenar.')
    return FlexAE(len(ae['features']), arch['encoder'], arch['decoder'])


# ── 6 · Ruido para denoising (segmento 06) ────────────────────────────────────
def noise(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    _need_scaled(frame)
    sigma, seed = float(p.get('sigma', .1)), int(p.get('seed', 42))
    if not 0 < sigma <= 2:
        raise ValueError('σ del ruido en (0, 2].')
    out = _out(frame, noise=sigma)
    xb = torch.tensor(_block(frame, 'TR')[:200])
    torch.manual_seed(seed)
    inp = core.add_noise(xb, sigma)
    snr = float(xb.var() / (inp - xb).var())
    f0 = out.attrs['ae']['features'][0]
    rows = [{'fila': i, 'limpio': float(a), 'con ruido': float(b)} for i, (a, b) in enumerate(zip(xb[:, 0], inp[:, 0]))]
    return out, _report('ae_noise', out, selected=f'gaussiano σ = {sigma}', scope='sólo entrada del DAE',
                        metrics={'σ': sigma, 'señal/ruido (var)': snr},
                        charts=[{'title': f'{f0}: entrada limpia vs contaminada (TR)', 'x_key': 'fila', 'series': ['limpio', 'con ruido'], 'rows': rows, 'chart_type': 'line'}],
                        notes=['inp = xb + σ·randn_like(xb) se aplica en cada lote; el objetivo sigue siendo xb limpio.'])


# ── 7 · Entrenar AE y DAE (segmento 07) ───────────────────────────────────────
def train(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _need_scaled(frame)
    which = [n for n, on in (('AE', p.get('train_ae', True)), ('DAE', p.get('train_dae', True))) if on]
    if not which:
        raise ValueError('Activa AE, DAE o ambos.')
    if 'DAE' in which and not ae.get('noise'):
        raise ValueError('El DAE necesita «6 · Ruido (denoising)» antes de entrenar.')
    dev = str(p.get('device', 'cpu'))
    device = core.get_device(None if dev == 'auto' else dev)
    cfg = dict(epochs=int(p.get('epochs', 80)), batch=int(p.get('batch_size', 64)), lr=float(p.get('lr', 1e-3)),
               patience=int(p.get('patience', 12)), grad_clip=float(p.get('grad_clip', 1.0)), seed=int(p.get('seed', 42)),
               weight_decay=float(p.get('weight_decay', 0.0)))
    if not 1 <= cfg['epochs'] <= 2000 or not 8 <= cfg['batch'] <= 4096 or not 0 < cfg['lr'] <= 1:
        raise ValueError('Épocas 1–2000, lote 8–4096 y lr en (0, 1].')
    xtr, xva = _block(frame, 'TR'), _block(frame, 'VA')
    models, table, curves = dict(ae.get('models', {})), [], {}
    for name in which:
        if progress:
            progress({'detail': f'Entrenando {name} en {core.describe_device(device)}'})
        tc = core.TrainConfig(**cfg, noise=ae['noise'] if name == 'DAE' else 0.0)
        try:
            model, hist = core.fit(build(frame), xtr, xtr, xva, xva, tc, device)
        except core.NumericalError as exc:
            raise ValueError(f'{name}: {exc}') from exc
        models[name] = _keep(model)
        curves[name] = hist.validation
        table.append({'modelo': name, 'ruido σ': tc.noise, 'épocas': len(hist.validation), 'mejor época': hist.best_epoch + 1,
                      'MSE TR final': float(hist.train[hist.best_epoch]), 'MSE VA mejor': float(hist.best_validation)})
    out = _out(frame, models=models)
    n = max(len(v) for v in curves.values())
    rows = [{'época': i + 1, **{k: float(v[i]) for k, v in curves.items() if i < len(v)}} for i in range(n)]
    return out, _report('ae_train', out, selected=' + '.join(which), scope=f'fit TR · early stopping VA · {core.describe_device(device)}',
                        table=table, metrics={f'MSE VA {r["modelo"]}': r['MSE VA mejor'] for r in table},
                        charts=[{'title': 'MSE de reconstrucción en VA por época', 'x_key': 'época', 'series': list(curves), 'rows': rows, 'chart_type': 'line'}],
                        notes=['Controles numéricos: entradas, pérdida y norma del gradiente finitas; clipping de gradiente; se restaura el mejor estado en VA.',
                               'Pesos iniciales y orden de lotes dependen sólo de la semilla: mismo resultado en cada ejecución (en CPU).'])


# ── 8 · Seleccionar por reconstrucción (segmento 08) ──────────────────────────
def select(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    comp = _compressors(frame)
    if not p.get('include_pca', True):
        comp.pop('PCA', None)
    if not comp:
        raise ValueError('No hay modelos: conecta «3 · Referencia PCA» y/o «7 · Entrenar AE / DAE».')
    comparison = core.compare_models(comp, _block(frame, 'VA'))
    selected = core.select_best(comparison)
    out = _out(frame, selected=selected)
    rows = comparison.to_dict(orient='records')
    return out, _report('ae_select', out, selected=selected, scope='menor MSE de reconstrucción en VA', table=rows,
                        metrics={r['modelo']: float(r['validation_reconstruction_MSE']) for r in rows},
                        charts=[{'title': 'MSE de reconstrucción en VA', 'x_key': 'modelo', 'series': ['validation_reconstruction_MSE'], 'rows': rows, 'chart_type': 'bar'}],
                        notes=[f'Seleccionado: {selected}. TE no participa en la elección.'])


# ── 9 · Evaluar utilidad con Ridge (segmento 09) ──────────────────────────────
def ridge(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _ae(frame)
    comp = _compressors(frame)
    if not comp:
        raise ValueError('No hay modelos que codifiquen: conecta PCA o Entrenar AE / DAE antes.')
    alpha = float(p.get('alpha', 1.0))
    if not 0 <= alpha <= 1e6:
        raise ValueError('α entre 0 y 1e6.')
    xtr, xva, ytr, yva = _block(frame, 'TR'), _block(frame, 'VA'), _y(frame, 'TR'), _y(frame, 'VA')
    downstream, table = {}, []
    for name, c in comp.items():
        m = Ridge(alpha=alpha).fit(c.encode(xtr), ytr)
        downstream[name] = _keep(m)
        pred = m.predict(c.encode(xva))
        table.append({'modelo': name, 'VA RMSE bps': float(mean_squared_error(yva, pred) ** .5), 'VA R2': float(r2_score(yva, pred))})
    base = float(np.sqrt(np.mean((yva - ytr.mean()) ** 2)))
    table.append({'modelo': 'media TR (línea base)', 'VA RMSE bps': base, 'VA R2': float(r2_score(yva, np.full_like(yva, ytr.mean())))})
    chosen = ae.get('selected') or next(iter(comp))
    data = frame.copy()
    z = comp[chosen].encode(frame[ae['features']].to_numpy(np.float32))
    for i in range(z.shape[1]):
        data[f'Z{i + 1}'] = z[:, i]
    data['Pred_bps'] = _get(downstream[chosen]).predict(z)
    out = _out(frame, data, ridge=downstream, alpha=alpha)
    return out, _report('ae_ridge', out, selected=f'Ridge(α={alpha}) sobre factores de {chosen}', scope='fit TR · medida VA', table=table,
                        metrics={r['modelo']: r['VA RMSE bps'] for r in table},
                        charts=[{'title': 'RMSE de y_bps en VA (bps)', 'x_key': 'modelo', 'series': ['VA RMSE bps'], 'rows': table, 'chart_type': 'bar'}],
                        notes=['Mide si los factores latentes sirven para predecir; reconstruir bien no garantiza predecir bien.',
                               f'Columnas añadidas: Z1…Z{z.shape[1]} y Pred_bps (modelo {chosen}).'])


# ── 10 · Medir ambas tareas en test (segmento 10) ─────────────────────────────
def test(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    ae = _ae(frame)
    comp = _compressors(frame)
    if not ae.get('ridge'):
        raise ValueError('Conecta «9 · Ridge sobre factores» antes de medir en test.')
    xte, yte = _block(frame, 'TE'), _y(frame, 'TE')
    rows = []
    for name, c in comp.items():
        y_hat = _get(ae['ridge'][name]).predict(c.encode(xte))
        rows.append({'modelo': name, 'test_reconstruction_MSE': core.reconstruction_mse(c.reconstruct(xte), xte),
                     'test_y_bps_RMSE': float(mean_squared_error(yte, y_hat) ** .5), 'test_y_bps_R2': float(r2_score(yte, y_hat))})
    chosen = ae.get('selected') or rows[0]['modelo']
    per = core.per_feature_mse(comp[chosen].reconstruct(xte), xte, ae['features'])
    te = frame[frame['Split'] == 'TE']
    charts = [{'title': f'MSE de reconstrucción por indicador ({chosen}, TE)', 'x_key': 'indicador', 'series': ['MSE'],
               'rows': [{'indicador': k, 'MSE': float(v)} for k, v in per.items()], 'chart_type': 'bar'},
              {'title': 'Reconstrucción vs predicción en TE', 'x_key': 'modelo', 'series': ['test_reconstruction_MSE'], 'rows': rows, 'chart_type': 'bar'}]
    if 'Pred_bps' in te:
        charts.append({'title': f'y_bps real vs predicho ({chosen}, TE)', 'x_key': 'x', 'series': ['real', 'predicho'], 'chart_type': 'line',
                       'rows': [{'x': _x(i), 'real': float(a), 'predicho': float(b)} for i, a, b in zip(te.index, te['y_bps'], te['Pred_bps'])][-1500:]})
    best_recon = min(rows, key=lambda r: r['test_reconstruction_MSE'])['modelo']
    best_pred = min(rows, key=lambda r: r['test_y_bps_RMSE'])['modelo']
    out = _out(frame)
    return out, _report('ae_test', out, selected=chosen, scope='TE · medido una vez', table=rows,
                        metrics={k: float(v) for k, v in next(r for r in rows if r['modelo'] == chosen).items() if k != 'modelo'},
                        charts=charts,
                        notes=[f'Mejor reconstrucción en TE: {best_recon}; mejor predicción de y_bps: {best_pred}.',
                               'Reconstrucción y predicción tienen criterios distintos: compara ambos antes de elegir representación.',
                               f'Indicador peor reconstruido: {per.idxmax()} (MSE {per.max():.4f}).'])


STEPS = {'ae_split': split, 'ae_standardize': standardize, 'ae_pca': pca, 'ae_encoder': encoder, 'ae_decoder': decoder,
         'ae_noise': noise, 'ae_train': train, 'ae_select': select, 'ae_ridge': ridge, 'ae_test': test}
