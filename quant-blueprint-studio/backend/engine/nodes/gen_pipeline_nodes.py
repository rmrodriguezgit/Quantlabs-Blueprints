"""Unidad 9 · Modelos generativos: diez herramientas independientes, una por segmento de la infografía.

01 Ventanas → 02 Condición causal → 03 Separar ventanas completas → 04 Escalar con TR →
05 CVAE → 06 Reparametrizar → 07 Reconstrucción + KL (entrena) → 08 cGAN (entrena) →
09 Seleccionar sin mirar test → 10 Evaluar escenarios y colas.

Como en el ML Pipeline, todo viaja por el pin ``frame``: el DataFrame diario pasa intacto (más columnas
informativas) y los metadatos en ``frame.attrs['gen']``. Arrays de ventanas, escaladores y redes se guardan
en un registro en proceso (pandas copia en profundidad los attrs); cada paso crea una entrada nueva que
comparte los arrays de la anterior, así una rama no altera a otra.

Reglas de fuga: la condición usa sólo las 20 sesiones anteriores; se descartan las ventanas que cruzan un
corte; escalas ajustadas con TR; los pesos se ajustan con TR; las instantáneas se comparan en VA con ruido
fijo; TE se mide una vez, en el paso 10.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .scaling_nodes import affine_audit

Progress = Callable[[dict[str, Any]], None] | None
_STORE: 'OrderedDict[str, dict[str, Any]]' = OrderedDict()
_MAX = 24
ORDER = ('gen_windows', 'gen_condition', 'gen_split', 'gen_scale', 'gen_cvae', 'gen_reparam',
         'gen_vae_train', 'gen_gan_train', 'gen_select', 'gen_evaluate')
LABEL = {'gen_windows': '01 · Crear ventanas', 'gen_condition': '02 · Condición causal', 'gen_split': '03 · Separar ventanas',
         'gen_scale': '04 · Escalar con TR', 'gen_cvae': '05 · CVAE', 'gen_reparam': '06 · Reparametrizar',
         'gen_vae_train': '07 · Reconstrucción + KL', 'gen_gan_train': '08 · cGAN', 'gen_select': '09 · Seleccionar',
         'gen_evaluate': '10 · Evaluar escenarios', 'gen_dcgan_train': '08 · cDCGAN 1D', 'gen_regime_compare': 'Comparar regímenes'}


# ── registro y utilidades ─────────────────────────────────────────────────────────────────────────
def _state(frame: pd.DataFrame, need: str | None = None) -> dict[str, Any]:
    gid = frame.attrs.get('gen', {}).get('id')
    st = _STORE.get(gid or '')
    if st is None:
        if need:
            raise ValueError(f'Falta «{LABEL[need]}» aguas arriba (o vuelve a ejecutar el blueprint completo).')
        return {}
    if need and need not in st['done']:
        raise ValueError(f'Falta «{LABEL[need]}» aguas arriba: conecta los pasos en el orden 01 → 10.')
    return st


def _emit(frame: pd.DataFrame, st: dict[str, Any], step: str, out: pd.DataFrame | None = None) -> pd.DataFrame:
    st = {**st, 'done': [*st.get('done', []), step]}
    gid = uuid.uuid4().hex[:12]
    _STORE[gid] = st
    while len(_STORE) > _MAX:
        _STORE.popitem(last=False)
    out = frame.copy() if out is None else out
    out.attrs = {**{k: v for k, v in frame.attrs.items() if k != 'gen'}, 'raw_rows': frame.attrs.get('raw_rows', len(frame)),
                 'gen': {'id': gid, 'steps': st['done'], **{k: st[k] for k in ('horizon', 'lookback', 'cuts') if k in st}}}
    return out


def _report(topic: str, **body: Any) -> dict[str, Any]:
    return {'study': True, 'topic': topic, 'selected': body.pop('selected', ''), 'features': body.pop('features', []),
            'splits': body.pop('splits', {'train': 0, 'validation': 0, 'test': 0}), 'metrics': body.pop('metrics', {}),
            'table': body.pop('table', []), 'charts': body.pop('charts', []), 'notes': body.pop('notes', []), **body}


def _chart(title, rows, series, x_key, chart_type='line'):
    return {'title': title, 'rows': rows, 'series': series, 'x_key': x_key, 'chart_type': chart_type}


def _x(v: Any) -> str:
    return v.isoformat()[:10] if hasattr(v, 'isoformat') else str(v)


def _splits(st):
    g = st.get('group')
    if g is None:
        return {'train': 0, 'validation': 0, 'test': 0}
    return {'train': int((g == 0).sum()), 'validation': int((g == 1).sum()), 'test': int((g == 2).sum())}


def _finite(name: str, v: torch.Tensor | float) -> None:
    t = v.detach() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    if not torch.isfinite(t).all():
        raise ValueError(f'Control numérico: {name} contiene NaN/inf. Baja el learning rate o revisa la escala (paso 04).')


def _device(p) -> torch.device:
    want = str(p.get('device', 'cpu'))
    if want == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if want == 'cuda' and not torch.cuda.is_available():
        raise ValueError('device = cuda pero no hay GPU disponible: usa cpu o auto.')
    return torch.device(want)


# ── 01 Crear ventanas de retornos ─────────────────────────────────────────────────────────────────
def windows(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    col = str(p.get('return_column', 'auto') or 'auto')
    if col == 'auto':
        if 'LogReturn' in frame:
            col, r = 'LogReturn', frame['LogReturn']
        elif 'Close' in frame:
            col, r = 'log(Close).diff()', np.log(frame['Close'].astype(float)).diff()
        else:
            raise ValueError('Sin columna de retornos: conecta una fuente con Close o elige return_column.')
    elif col in frame:
        r = frame[col]
    else:
        raise ValueError(f'Columna «{col}» inexistente. Disponibles: {", ".join(map(str, frame.columns))}')
    r = pd.to_numeric(r, errors='coerce').astype(float).dropna()
    if not np.isfinite(r.to_numpy()).all():
        raise ValueError('Retornos con infinitos: conecta «Limpieza de datos» antes.')
    if r.abs().max() > 1:
        raise ValueError(f'|{col}| > 1: parecen precios o porcentajes, no retornos log.')
    if r.std() == 0:
        raise ValueError('Serie de retornos constante.')
    h, stride = int(p.get('horizon', 16)), int(p.get('stride', 1))
    if not 4 <= h <= 64:
        raise ValueError('HORIZON debe estar entre 4 y 64 sesiones.')
    starts = np.arange(0, len(r) - h + 1, stride)
    if len(starts) < 60:
        raise ValueError(f'Sólo {len(starts)} ventanas: amplía la historia o reduce HORIZON.')
    ret = r.to_numpy(np.float64)
    W = np.stack([ret[i:i + h] for i in starts]).astype(np.float32)
    st = {'ret': ret, 'dates': r.index, 'starts': starts, 'W': W, 'horizon': h, 'return_column': col}
    out = frame.copy()
    out['Gen_WindowStart'] = out.index.isin(r.index[starts]).astype(int)
    ex = [{'sesión': k + 1, 'retorno': float(W[0, k])} for k in range(h)]
    return _emit(frame, st, 'gen_windows', out), _report(
        'gen_windows', selected=f'HORIZON = {h} · paso {stride}', features=[col],
        metrics={'ventanas': len(W), 'sesiones': len(ret), 'horizon': h},
        table=[{'ventana': i, 'inicio': _x(r.index[s]), 'fin': _x(r.index[s + h - 1])} for i, s in enumerate(starts[:5])],
        charts=[_chart('Primera ventana · w = ret[i:i+HORIZON]', ex, ['retorno'], 'sesión', 'bar')],
        notes=[f'Cada muestra contiene {h} sesiones consecutivas de «{col}».',
               'Paso 1 = ventanas solapadas (más muestras, correlacionadas); paso = HORIZON = sin solape.'])


# ── 02 Definir una condición causal ───────────────────────────────────────────────────────────────
def condition(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_windows')
    lb, floor = int(p.get('lookback', 20)), float(p.get('floor', 1e-8))
    kinds = [k for k in ('log_vol', 'momentum') if p.get('use_' + k, k == 'log_vol')]
    if not kinds:
        raise ValueError('Activa al menos una condición (log_vol o momentum).')
    ret, starts = st['ret'], st['starts']
    keep = starts >= lb
    starts = starts[keep]
    C = []
    for i in starts:
        past = ret[i - lb:i]
        row = []
        if 'log_vol' in kinds:
            row.append(np.log(max(np.std(past, ddof=1), floor)))
        if 'momentum' in kinds:
            row.append(past.sum())
        C.append(row)
    C = np.asarray(C, np.float32)
    st = {**st, 'starts': starts, 'W': st['W'][keep], 'C': C, 'cond_names': kinds, 'lookback': lb}
    out = frame.copy()
    vol = pd.Series(ret, index=st['dates']).rolling(lb).std(ddof=1).shift(0)
    out['Gen_CondVol'] = np.log(vol.reindex(out.index).clip(lower=floor))
    rows = [{'date': _x(st['dates'][s]), 'log_vol': float(c[0])} for s, c in zip(starts[::max(len(starts) // 300, 1)], C[::max(len(starts) // 300, 1)])] if 'log_vol' in kinds else []
    return _emit(frame, st, 'gen_condition', out), _report(
        'gen_condition', selected=' + '.join(kinds), features=kinds,
        metrics={'ventanas': len(C), 'descartadas_sin_historia': int((~keep).sum()), 'lookback': lb},
        charts=[_chart(f'Condición c = log(std(ret[i-{lb}:i]))', rows, ['log_vol'], 'date')] if rows else [],
        notes=[f'La condición sólo usa las {lb} sesiones anteriores al inicio de la ventana: nunca ve la ventana que se va a generar.',
               'log(max(σ, floor)) evita log(0) en tramos sin movimiento. Gen_CondVol (columna) es la misma cantidad por fecha.'])


# ── 03 Separar ventanas completas ─────────────────────────────────────────────────────────────────
def split(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_condition')
    dates, starts, h = st['dates'], st['starts'], st['horizon']
    start, end = dates[starts], dates[starts + h - 1]
    if str(p.get('cut_mode', 'ratio')) == 'dates':
        vs, ts = pd.Timestamp(str(p.get('val_start'))), pd.Timestamp(str(p.get('test_start')))
    else:
        tr, va = float(p.get('train_ratio', .7)), float(p.get('val_ratio', .15))
        if tr + va >= .95:
            raise ValueError('TR + VA debe ser < 0.95 para dejar ventanas de prueba.')
        vs, ts = dates[int(tr * len(dates))], dates[int((tr + va) * len(dates))]
    if not vs < ts:
        raise ValueError('VAL_START debe ser anterior a TEST_START.')
    train = end < vs
    valid = (start >= vs) & (end < ts)
    test = start >= ts
    g = np.where(train, 0, np.where(valid, 1, np.where(test, 2, -1)))
    counts = {k: int((g == i).sum()) for i, k in enumerate(('train', 'validation', 'test'))}
    if min(counts.values()) < 8:
        raise ValueError(f'Cada bloque necesita ≥ 8 ventanas completas ({counts}). Amplía la historia o ajusta los cortes.')
    keep = g >= 0
    st = {**st, 'starts': starts[keep], 'W': st['W'][keep], 'C': st['C'][keep], 'group': g[keep],
          'cuts': {'VAL_START': _x(vs), 'TEST_START': _x(ts)}}
    out = frame.copy()
    out['Split'] = np.where(out.index < vs, 'TR', np.where(out.index < ts, 'VA', 'TE'))
    return _emit(frame, st, 'gen_split', out), _report(
        'gen_split', selected=f'VAL_START {_x(vs)} · TEST_START {_x(ts)}', splits=counts,
        metrics={**counts, 'purgadas': int((~keep).sum())},
        table=[{'bloque': k, 'ventanas': v} for k, v in counts.items()] + [{'bloque': 'purgadas (cruzan un corte)', 'ventanas': int((~keep).sum())}],
        notes=['train = gd.end < VAL_START · valid = (gd.start ≥ VAL_START) & (gd.end < TEST_START) · test = gd.start ≥ TEST_START.',
               'Las ventanas que cruzan un corte se eliminan: ninguna muestra de entrenamiento contiene sesiones de VA/TE.'])


# ── 04 Escalar con entrenamiento ──────────────────────────────────────────────────────────────────
def scale(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_split')
    tr = st['group'] == 0
    W, C = st['W'], st['C']
    rm, rs = float(W[tr].mean()), max(float(W[tr].std()), 1e-12)
    if str(p.get('return_scale', 'global')) == 'per_step':
        rm_v, rs_v = W[tr].mean(0), np.maximum(W[tr].std(0), 1e-12)
    else:
        rm_v, rs_v = np.full(W.shape[1], rm, np.float32), np.full(W.shape[1], rs, np.float32)
    cm, cs = C[tr].mean(0), np.maximum(C[tr].std(0), 1e-12)
    GX = ((W - rm_v) / rs_v).astype(np.float32)
    GC = ((C - cm) / cs).astype(np.float32)
    sc = {'return_mean': rm_v.astype(np.float32), 'return_std': rs_v.astype(np.float32), 'cond_mean': cm, 'cond_std': cs}
    st = {**st, 'GX': GX, 'GC': GC, 'scaler': sc}
    audits = [affine_audit(W[tr].ravel(), rm, rs, 'Ventanas de retornos (TR)', method='Media/σ TR (escala global)', inverse=True)]
    for j, n in enumerate(st['cond_names']):
        audits.append(affine_audit(C[tr, j], float(cm[j]), float(cs[j]), f'Condición {n} (TR)', method='Media/σ TR', inverse=False))
    tab = [{'escala': 'return_mean', 'valor': rm}, {'escala': 'return_std', 'valor': rs}] + \
          [{'escala': f'{n}_mean / std', 'valor': f'{cm[j]:.4f} / {cs[j]:.4f}'} for j, n in enumerate(st['cond_names'])]
    va = st['group'] == 1
    return _emit(frame, st, 'gen_scale'), _report(
        'gen_scale', selected=str(p.get('return_scale', 'global')), splits=_splits(st), table=tab, scaling_audits=audits,
        metrics={'return_mean': rm, 'return_std': rs, 'media_VA_escalada': float(GX[va].mean()), 'σ_VA_escalada': float(GX[va].std())},
        notes=['return_mean = gw[gm["train"]].mean() · return_std = gw[gm["train"]].std(): una escala para retornos y otra para la condición.',
               'VA y TE se transforman con las escalas de TR; por eso su media/σ escaladas no son exactamente 0/1.'])


# ── 05 Construir el VAE condicional ───────────────────────────────────────────────────────────────
def _mlp(d_in: int, widths: list[int], d_out: int, act: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for w in widths:
        layers += [nn.Linear(d_in, w), act()]
        d_in = w
    return nn.Sequential(*layers, nn.Linear(d_in, d_out))


def _widths(v: Any) -> list[int]:
    w = [int(s) for s in str(v or '64,64').split(',') if s.strip()]
    if not w or any(x < 2 or x > 1024 for x in w) or len(w) > 6:
        raise ValueError('Anchos ocultos: de 1 a 6 enteros entre 2 y 1024 separados por coma.')
    return w


class CVAE(nn.Module):
    def __init__(self, horizon: int, zdim: int, cdim: int, widths: list[int], clamp: float = 8.):
        super().__init__()
        self.zdim, self.clamp, self.temperature, self.sample = zdim, clamp, 1.0, True
        self.enc = _mlp(horizon + cdim, widths[:-1], widths[-1], nn.ReLU)
        self.mu, self.logvar = nn.Linear(widths[-1], zdim), nn.Linear(widths[-1], zdim)
        self.dec = _mlp(zdim + cdim, widths[::-1], horizon, nn.ReLU)

    def encode(self, x, c):
        h = F.relu(self.enc(torch.cat([x, c], 1)))
        return self.mu(h), self.logvar(h).clamp(-self.clamp, self.clamp)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(.5 * lv) * torch.randn_like(mu) if self.sample else mu

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        return self.dec(torch.cat([self.reparameterize(mu, lv), c], 1)), mu, lv

    def generate(self, z, c):
        return self.dec(torch.cat([z * self.temperature, c], 1))


def cvae(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_scale')
    zdim, widths, clamp = int(p.get('latent_dim', 4)), _widths(p.get('widths', '64,64')), float(p.get('logvar_clamp', 8))
    h, cd = st['GX'].shape[1], st['GC'].shape[1]
    net = CVAE(h, zdim, cd, widths, clamp)
    n = sum(t.numel() for t in net.parameters())
    arch = {'horizon': h, 'zdim': zdim, 'cdim': cd, 'widths': widths, 'clamp': clamp}
    st = {**st, 'cvae_arch': arch, 'zdim': zdim}
    return _emit(frame, st, 'gen_cvae'), _report(
        'gen_cvae', selected=f'[{h}+{cd}] → {"→".join(map(str, widths))} → μ, log σ² ({zdim})', splits=_splits(st),
        metrics={'parametros': n, 'ZDIM': zdim}, features=st['cond_names'],
        table=[{'bloque': 'Encoder', 'forma': f'concat(x[{h}], c[{cd}]) → {" → ".join(map(str, widths))}'},
               {'bloque': 'μ / log σ²', 'forma': f'{widths[-1]} → {zdim} · clamp(±{clamp:g})'},
               {'bloque': 'Decoder', 'forma': f'concat(z[{zdim}], c[{cd}]) → {" → ".join(map(str, widths[::-1]))} → {h}'}],
        notes=['h = enc(cat([x, c], 1)) · mu = mu(h) · lv = logvar(h).clamp(-8, 8): el clamp evita exp(lv) = inf.',
               'La red se construye aquí y se entrena en «07 · Reconstrucción + KL».'])


# ── 06 Reparametrizar ─────────────────────────────────────────────────────────────────────────────
def reparam(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_cvae')
    mode, temp = str(p.get('mode', 'sample')), float(p.get('temperature', 1.0))
    if not 0 < temp <= 3:
        raise ValueError('La temperatura debe estar en (0, 3].')
    g = torch.Generator().manual_seed(int(p.get('seed', 0)))
    mu, lv = torch.tensor([[.5, -1.]]).repeat(4000, 1), torch.tensor([[0., -2.]]).repeat(4000, 1)
    eps = torch.randn(mu.shape, generator=g)
    z = mu + torch.exp(.5 * lv) * eps
    edges = np.linspace(-3, 3, 31)
    he, _ = np.histogram(eps[:, 0].numpy(), edges, density=True)
    hz, _ = np.histogram(z[:, 0].numpy(), edges, density=True)
    rows = [{'valor': float((edges[i] + edges[i + 1]) / 2), 'ε ~ N(0,1)': float(he[i]), 'z = μ + σ ε (μ=.5, σ=1)': float(hz[i])} for i in range(30)]
    st = {**st, 'reparam': {'sample': mode == 'sample', 'temperature': temp}}
    return _emit(frame, st, 'gen_reparam'), _report(
        'gen_reparam', selected=f'{mode} · temperatura {temp:g}', splits=_splits(st),
        metrics={'media_z_dim2': float(z[:, 1].mean()), 'σ_z_dim2': float(z[:, 1].std()), 'σ_esperada_dim2': float(np.exp(-1))},
        charts=[_chart('Reparametrización: ruido aleatorio vs z', rows, list(rows[0])[1:], 'valor', 'bar')],
        notes=['z = mu + exp(.5·lv) · randn_like(mu): el ruido queda fuera del grafo y el gradiente fluye por μ y log σ².',
               'mode = mean usa z = μ (ablación sin muestreo: el modelo degenera en autoencoder).',
               'La temperatura multiplica z ~ N(0, I) al generar escenarios (09 y 10); < 1 concentra, > 1 dispersa.'])


# ── entrenamiento compartido ──────────────────────────────────────────────────────────────────────
def _batches(n, b, g):
    return torch.randperm(n, generator=g).split(b)


def _snap(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# ── 07 Equilibrar reconstrucción y KL (entrena el CVAE) ───────────────────────────────────────────
def vae_train(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_reparam')
    seed, dev = int(p.get('seed', 42)), _device(p)
    torch.manual_seed(seed)
    a = st['cvae_arch']
    net = CVAE(a['horizon'], a['zdim'], a['cdim'], a['widths'], a['clamp'])
    net.sample, net.temperature = st['reparam']['sample'], st['reparam']['temperature']
    net = net.to(dev)
    tr = st['group'] == 0
    x, c = torch.tensor(st['GX'][tr], device=dev), torch.tensor(st['GC'][tr], device=dev)
    beta, warm = float(p.get('beta', .01)), int(p.get('kl_warmup', 0))
    epochs, bs = int(p.get('epochs', 60)), int(p.get('batch_size', 128))
    opt = torch.optim.Adam(net.parameters(), lr=float(p.get('lr', 1e-3)))
    g, hist, snaps, t0 = torch.Generator().manual_seed(seed), [], [], time.perf_counter()
    every = max(int(p.get('snapshot_every', 1)), 1)
    for ep in range(epochs):
        net.train()
        b_eff = beta * min(1., (ep + 1) / warm) if warm else beta
        tot = [0., 0., 0.]
        for idx in _batches(len(x), bs, g):
            idx = idx.to(dev)
            recon, mu, lv = net(x[idx], c[idx])
            mse = F.mse_loss(recon, x[idx])
            kl = .5 * (mu.square() + lv.exp() - 1 - lv).sum(1).mean()
            loss = mse + b_eff * kl
            _finite('pérdida CVAE', loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            _finite('norma del gradiente', nn.utils.clip_grad_norm_(net.parameters(), float(p.get('grad_clip', 1.0))))
            opt.step()
            for k, v in enumerate((loss, mse, kl)):
                tot[k] += float(v.detach()) * len(idx)
        hist.append({'epoch': ep + 1, 'total': tot[0] / len(x), 'MSE': tot[1] / len(x), 'KL': tot[2] / len(x), 'β·KL': b_eff * tot[2] / len(x)})
        if (ep + 1) % every == 0 or ep == epochs - 1:
            snaps.append((ep + 1, _snap(net)))
        if progress:
            progress({'epoch': ep + 1, 'epochs': epochs, 'train': hist[-1]['total'], 'detail': 'CVAE'})
    net = net.cpu().eval()
    models = {**st.get('models', {}), 'CVAE': {'net': net, 'snaps': snaps, 'kind': 'cvae'}}
    st = {**st, 'models': models}
    last = hist[-1]
    return _emit(frame, st, 'gen_vae_train'), _report(
        'gen_vae_train', selected=f'β = {beta:g}' + (f' · calentamiento {warm} épocas' if warm else ''), splits=_splits(st),
        metrics={'MSE_final': last['MSE'], 'KL_final': last['KL'], 'segundos': time.perf_counter() - t0, 'instantaneas': len(snaps)},
        charts=[_chart('Reconstrucción MSE y KL por época (TR)', hist, ['MSE', 'KL', 'β·KL'], 'epoch')],
        notes=['kl = .5 · (μ² + exp(lv) − 1 − lv).sum(1).mean() · loss = mse + β·kl.',
               'β bajo → copia ventanas (KL alto, latente poco útil para muestrear); β alto → colapso posterior (KL ≈ 0, escenarios planos).',
               f'Se guardan {len(snaps)} instantáneas; «09 · Seleccionar» elige la mejor en VA. Aquí no se mira VA ni TE.'])


# ── 08 Alternar generador y discriminador ─────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, horizon, zdim, cdim, widths):
        super().__init__()
        self.zdim, self.temperature = zdim, 1.0
        self.net = _mlp(zdim + cdim, widths, horizon, lambda: nn.LeakyReLU(.2))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))

    def generate(self, z, c):
        return self(z * self.temperature, c)


def gan_train(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_scale')
    seed, dev = int(p.get('seed', 42)), _device(p)
    torch.manual_seed(seed)
    zdim = int(p.get('latent_dim', st.get('zdim', 4)))
    h, cd = st['GX'].shape[1], st['GC'].shape[1]
    w = _widths(p.get('widths', '64,64'))
    gen = Generator(h, zdim, cd, w).to(dev)
    disc = _mlp(h + cd, w, 1, lambda: nn.LeakyReLU(.2)).to(dev)
    betas = (.5, .999)
    og = torch.optim.Adam(gen.parameters(), lr=float(p.get('lr_g', 2e-4)), betas=betas)
    od = torch.optim.Adam(disc.parameters(), lr=float(p.get('lr_d', 2e-4)), betas=betas)
    tr = st['group'] == 0
    x, c = torch.tensor(st['GX'][tr], device=dev), torch.tensor(st['GC'][tr], device=dev)
    epochs, bs, d_steps = int(p.get('epochs', 60)), int(p.get('batch_size', 128)), int(p.get('d_steps', 1))
    smooth, clip = float(p.get('label_smoothing', 0.)), float(p.get('grad_clip', 1.0))
    amp = bool(p.get('amp', False)) and dev.type == 'cuda'
    every = max(int(p.get('snapshot_every', 1)), 1)
    g, hist, snaps, t0 = torch.Generator().manual_seed(seed), [], [], time.perf_counter()
    bce = F.binary_cross_entropy_with_logits
    for ep in range(epochs):
        gen.train(); disc.train()
        dl_s = gl_s = dr_s = df_s = 0.
        for idx in _batches(len(x), bs, g):
            idx = idx.to(dev)
            xb, cb = x[idx], c[idx]
            for _ in range(d_steps):
                with torch.autocast(dev.type, enabled=amp):
                    fake = gen(torch.randn(len(xb), zdim, device=dev), cb)
                    lr_ = disc(torch.cat([xb, cb], 1))
                    lf_ = disc(torch.cat([fake.detach(), cb], 1))
                dl = bce(lr_.float(), torch.full_like(lr_, 1 - smooth).float()) + bce(lf_.float(), torch.zeros_like(lf_).float())
                _finite('pérdida D', dl)
                od.zero_grad(set_to_none=True); dl.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), clip); od.step()
            with torch.autocast(dev.type, enabled=amp):
                fake = gen(torch.randn(len(xb), zdim, device=dev), cb)
                logits = disc(torch.cat([fake, cb], 1))
            gl = bce(logits.float(), torch.ones_like(logits).float())
            _finite('pérdida G', gl)
            og.zero_grad(set_to_none=True); gl.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), clip); og.step()
            n = len(idx)
            dl_s += float(dl.detach()) * n; gl_s += float(gl.detach()) * n
            dr_s += float(torch.sigmoid(lr_.detach().float()).mean()) * n; df_s += float(torch.sigmoid(lf_.detach().float()).mean()) * n
        hist.append({'epoch': ep + 1, 'D': dl_s / len(x), 'G': gl_s / len(x), 'D(real)': dr_s / len(x), 'D(sintético)': df_s / len(x)})
        if (ep + 1) % every == 0 or ep == epochs - 1:
            snaps.append((ep + 1, _snap(gen)))
        if progress:
            progress({'epoch': ep + 1, 'epochs': epochs, 'train': hist[-1]['G'], 'detail': 'cGAN'})
    gen = gen.cpu().eval()
    gen.temperature = st.get('reparam', {}).get('temperature', 1.0)
    models = {**st.get('models', {}), 'cGAN': {'net': gen, 'snaps': snaps, 'kind': 'gan'}}
    st = {**st, 'models': models, 'zdim': st.get('zdim', zdim)}
    last = hist[-1]
    warn = []
    if last['D(real)'] > .95 and last['D(sintético)'] < .05:
        warn.append('D gana con holgura (D(real)≈1, D(sintético)≈0): baja lr_d, sube label_smoothing o d_steps = 1.')
    return _emit(frame, st, 'gen_gan_train'), _report(
        'gen_gan_train', selected=f'G/D {"→".join(map(str, w))} · d_steps {d_steps}' + (' · AMP' if amp else ''), splits=_splits(st),
        metrics={'D_final': last['D'], 'G_final': last['G'], 'D(real)': last['D(real)'], 'D(sintético)': last['D(sintético)'],
                 'segundos': time.perf_counter() - t0, 'instantaneas': len(snaps)},
        charts=[_chart('Pérdidas alternadas D / G (TR)', hist, ['D', 'G'], 'epoch'),
                _chart('Salida del discriminador', hist, ['D(real)', 'D(sintético)'], 'epoch')],
        notes=['fake = gen(z, cb) · logits = disc(fake, cb) · gl = bce(logits.float(), ones_like(logits).float()).',
               'Las BCE se calculan en float32 aunque el forward use AMP (float16) en GPU.',
               'Las pérdidas de una GAN no indican calidad; la selección se hace en «09» con métricas de distribución en VA.', *warn])


# ── métricas de escenarios ────────────────────────────────────────────────────────────────────────
def _acf1(a):
    a = a - a.mean(1, keepdims=True)
    den = (a ** 2).sum(1)
    ok = den > 0
    return float(((a[:, 1:] * a[:, :-1]).sum(1)[ok] / den[ok]).mean()) if ok.any() else 0.


def _w1(a, b):
    q = np.linspace(0, 1, 201)[1:-1]
    return float(np.mean(np.abs(np.quantile(np.ravel(a), q) - np.quantile(np.ravel(b), q))))


def _tail_dep(W, q=.1):
    """P(r_{t+1} en cola inferior | r_t en cola inferior), umbral común."""
    a, b = W[:, :-1].ravel(), W[:, 1:].ravel()
    t = np.quantile(W, q)
    m = a <= t
    return float((b[m] <= t).mean()) if m.any() else 0.


def _stats(W, level=.05):
    r, cum = W.ravel(), W.sum(1)
    qq = np.quantile(r, level)
    sd = r.std() or 1e-12
    return {'media': float(r.mean()), 'σ': float(r.std()), 'asimetría': float(((r - r.mean()) ** 3).mean() / sd ** 3),
            'curtosis': float(((r - r.mean()) ** 4).mean() / sd ** 4), 'VaR': float(-qq), 'ES': float(-r[r <= qq].mean()),
            'σ_acum': float(cum.std()), 'VaR_acum': float(-np.quantile(cum, level)),
            'ACF1_r': _acf1(W), 'ACF1_|r|': _acf1(np.abs(W)), 'dep_cola': _tail_dep(W)}


def _score(fake, real, w):
    sf, sr = _stats(fake), _stats(real)
    return (w[0] * _w1(fake, real) + w[1] * _w1(fake.sum(1), real.sum(1)) / np.sqrt(fake.shape[1])
            + w[2] * abs(sf['ACF1_|r|'] - sr['ACF1_|r|']) + w[3] * abs(sf['ES'] - sr['ES']))


def _net_from(spec, state, st):
    net = deepcopy(spec['net'])
    net.load_state_dict(state)
    return net.eval()


# ── 09 Seleccionar sin mirar test ─────────────────────────────────────────────────────────────────
def select(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_scale')
    models = st.get('models') or {}
    if not models:
        raise ValueError('No hay modelos entrenados: conecta «07 · Reconstrucción + KL» y/o «08 · cGAN» antes.')
    va = st['group'] == 1
    real, cv = st['GX'][va], torch.tensor(st['GC'][va])
    seed = int(p.get('noise_seed', 43))
    w = [float(p.get(k, d)) for k, d in (('w_marginal', 1), ('w_cumulative', 1), ('w_vol_cluster', 1), ('w_tail', 1))]
    rows, best, curves = [], {}, {}
    for name, spec in models.items():
        fixed_z = torch.randn(len(real), spec['net'].zdim, generator=torch.Generator().manual_seed(seed))
        for ep, state in spec['snaps']:
            with torch.no_grad():
                fake = _net_from(spec, state, st).generate(fixed_z, cv).numpy()
            s = _score(fake, real, w)
            rows.append({'modelo': name, 'época': ep, 'score_VA': s})
            curves.setdefault(ep, {'epoch': ep})[name] = s
            if s < best.get(name, (np.inf,))[0]:
                best[name] = (s, ep, state)
    sel = min(best, key=lambda k: best[k][0])
    chosen = {n: {**models[n], 'net': _net_from(models[n], best[n][2], st), 'best_epoch': best[n][1], 'score': best[n][0]} for n in best}
    st = {**st, 'chosen': chosen, 'selected': sel, 'score_weights': w}
    table = [{'modelo': n, 'mejor época': best[n][1], 'score VA (menor = mejor)': best[n][0], 'seleccionado': n == sel} for n in best]
    return _emit(frame, st, 'gen_select'), _report(
        'gen_select', selected=f'{sel} · época {best[sel][1]}', splits=_splits(st),
        metrics={f'score_VA_{n}': best[n][0] for n in best}, table=table,
        charts=[_chart('Score de validación con ruido fijo', sorted(curves.values(), key=lambda r: r['epoch']), list(best), 'epoch')],
        notes=[f'fixed_z = torch.randn(len(GX["validation"]), ZDIM): el mismo ruido (semilla {int(p.get("noise_seed", 43))}) para todas las épocas y modelos.',
               'Score = w₁·W1 marginal + w₂·W1 del retorno acumulado/√h + w₃·|ΔACF1 de |r|| + w₄·|ΔES5 %|, en unidades escaladas.',
               'TE no se usa aquí: se mide una sola vez en «10 · Evaluar escenarios».'])


# ── 10 Evaluar escenarios y colas ─────────────────────────────────────────────────────────────────
def evaluate(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_select')
    te = st['group'] == 2
    sc = st['scaler']
    level, n_per = float(p.get('var_level', .05)), int(p.get('scenarios_per_condition', 20))
    inv = lambda a: a * sc['return_std'] + sc['return_mean']
    real = st['W'][te]
    cond = torch.tensor(st['GC'][te]).repeat_interleave(n_per, 0)
    zf = lambda d, s: torch.randn(len(cond) if s is None else s, d, generator=torch.Generator().manual_seed(int(p.get('seed', 7))))
    fakes = {}
    with torch.no_grad():
        for n, spec in st['chosen'].items():
            fakes[n] = inv(spec['net'].generate(zf(spec['net'].zdim, None), cond).numpy())
    if p.get('bootstrap_baseline', True):
        trW = st['W'][st['group'] == 0]
        fakes['Bootstrap TR'] = trW[np.random.default_rng(int(p.get('seed', 7))).integers(0, len(trW), len(cond))]
    stats = {'Real TE': _stats(real, level), **{k: _stats(v, level) for k, v in fakes.items()}}
    sel = st['selected']
    tab = [{'fuente': k, **{m: round(v, 6) for m, v in s.items()}, 'W1 vs real': (0. if k == 'Real TE' else _w1(fakes[k], real))} for k, s in stats.items()]
    edges = np.quantile(np.r_[real.ravel(), fakes[sel].ravel()], [.002, .998])
    edges = np.linspace(edges[0], edges[1], 31)
    hr, _ = np.histogram(real.ravel(), edges, density=True)
    hf, _ = np.histogram(fakes[sel].ravel(), edges, density=True)
    hist = [{'retorno': float((edges[i] + edges[i + 1]) / 2), 'real': float(hr[i]), sel: float(hf[i])} for i in range(30)]
    rr, ff = real[:, :-1].ravel(), fakes[sel][:, :-1].ravel()
    k = min(400, len(rr), len(ff))
    ir = np.random.default_rng(0).choice(len(rr), k, replace=False)
    jf = np.random.default_rng(1).choice(len(ff), k, replace=False)
    dep = [{'|r_t|': float(abs(rr[i])), 'real |r_t+1|': float(abs(real[:, 1:].ravel()[i]))} for i in ir] + \
          [{'|r_t|': float(abs(ff[j])), f'{sel} |r_t+1|': float(abs(fakes[sel][:, 1:].ravel()[j]))} for j in jf]
    paths = [{'sesión': t + 1, 'real': float(np.cumsum(real[0])[t]), **{f'{sel} #{i + 1}': float(np.cumsum(fakes[sel][i])[t]) for i in range(min(5, n_per))}} for t in range(real.shape[1])]
    # regímenes: la condición debe mover la dispersión
    reg = []
    if 'log_vol' in st['cond_names']:
        j = st['cond_names'].index('log_vol')
        for q in (.1, .5, .9):
            cq = np.quantile(st['C'][st['group'] == 0, j], q)
            cc = torch.tensor(st['GC'][te][:1]).repeat(2000, 1)
            cc[:, j] = float((cq - sc['cond_mean'][j]) / sc['cond_std'][j])
            zz = torch.randn(2000, st['chosen'][sel]['net'].zdim, generator=torch.Generator().manual_seed(11))
            with torch.no_grad():
                s = inv(st['chosen'][sel]['net'].generate(zz, cc).numpy())
            reg.append({'cuantil vol TR': q, 'σ condición': float(np.exp(cq)), 'σ generada': float(s.std()), f'VaR{int(level * 100)} {real.shape[1]}d': float(-np.quantile(s.sum(1), level))})
    # salida: por fecha de inicio de ventana TE, VaR del retorno acumulado generado
    out = frame.copy()
    starts = st['dates'][st['starts'][te]]
    var_cum = -np.quantile(fakes[sel].sum(1).reshape(-1, n_per), level, axis=1)
    out['Gen_VaR_cum'] = pd.Series(var_cum, index=starts).reindex(out.index)
    out['Real_cum'] = pd.Series(real.sum(1), index=starts).reindex(out.index)
    breaches = float((real.sum(1) < -var_cum).mean())
    charts = [_chart('Distribución de retornos diarios · TE', hist, ['real', sel], 'retorno', 'bar'),
              _chart('Dependencia de colas · |r_t| vs |r_t+1|', dep, ['real |r_t+1|', f'{sel} |r_t+1|'], '|r_t|', 'scatter'),
              _chart('Primera ventana TE · retornos acumulados', paths, list(paths[0])[1:], 'sesión'),
              _chart(f'VaR {int(level * 100)} % generado vs retorno acumulado real', [{'date': _x(d), 'Real_cum': float(r), '−VaR': float(-v)} for d, r, v in zip(starts, real.sum(1), var_cum)], ['Real_cum', '−VaR'], 'date')]
    if reg:
        charts.append(_chart('Condición → dispersión generada', reg, ['σ condición', 'σ generada'], 'cuantil vol TR', 'bar'))
    return _emit(frame, st, 'gen_evaluate', out), _report(
        'gen_evaluate', selected=sel, splits=_splits(st), table=tab + [{'fuente': f'régimen q={r["cuantil vol TR"]}', **r} for r in reg],
        metrics={'W1_TE': _w1(fakes[sel], real), f'excesos_VaR{int(level * 100)}_acum': breaches, 'esperado': level,
                 **{f'{m}_real': stats['Real TE'][m] for m in ('σ', 'ES', 'ACF1_|r|', 'dep_cola')},
                 **{f'{m}_{sel}': stats[sel][m] for m in ('σ', 'ES', 'ACF1_|r|', 'dep_cola')}},
        charts=charts,
        notes=[f'with torch.no_grad(): scenarios = model.generate(z, tc) — {n_per} escenarios por ventana TE con su condición real.',
               'Bootstrap TR = ventanas reales de entrenamiento al azar: línea base que ignora la condición. Un generador útil debe batirla en colas/dependencia.',
               f'Excesos: fracción de ventanas TE cuyo retorno acumulado real cae bajo −VaR generado (esperado ≈ {level:g}).',
               'Escenarios hipotéticos: no son intervalos de predicción calibrados.'])


# ── Ampliación cDCGAN 1D (infografía «unidad-9-cdcgan») ───────────────────────────────────────────
REGIMES = ('Bajista', 'Lateral', 'Alcista')


def _regime(c: torch.Tensor, j: int, cut: float) -> torch.Tensor:
    """Régimen causal desde el momentum escalado (TR): < −cut bajista, > cut alcista."""
    m = c[:, j]
    return torch.where(m < -cut, 0, torch.where(m > cut, 2, 1)).long()


class DCGenerator(nn.Module):
    """[z | condición | embedding de régimen] → Dense → reshape → ConvTranspose1d ×2 → retornos lineales."""

    def __init__(self, horizon: int, zdim: int, cdim: int, emb: int, mom: int, cut: float, ch: int, batchnorm: bool):
        super().__init__()
        if horizon % 4:
            raise ValueError('La cDCGAN necesita HORIZON múltiplo de 4 (dos ConvTranspose1d de stride 2).')
        self.zdim, self.temperature, self.mom, self.cut, self.ch, self.base = zdim, 1.0, mom, cut, ch, horizon // 4
        self.regime = nn.Embedding(3, emb)
        self.dense = nn.Sequential(nn.Linear(zdim + cdim + emb, ch * self.base), nn.LeakyReLU(.2))
        norm = (lambda n: nn.BatchNorm1d(n)) if batchnorm else (lambda n: nn.Identity())
        self.up = nn.Sequential(nn.ConvTranspose1d(ch, ch // 2, 4, 2, 1), norm(ch // 2), nn.LeakyReLU(.2), nn.ConvTranspose1d(ch // 2, 1, 4, 2, 1))

    def forward(self, z, c):
        h = self.dense(torch.cat([z, c, self.regime(_regime(c, self.mom, self.cut))], 1))
        return self.up(h.reshape(-1, self.ch, self.base)).squeeze(1)

    def generate(self, z, c):
        return self(z * self.temperature, c)


class DCDiscriminator(nn.Module):
    """[ventana | condición difundida en el tiempo] → Conv1d ×2 → logit."""

    def __init__(self, horizon: int, cdim: int, emb: int, mom: int, cut: float, ch: int):
        super().__init__()
        self.mom, self.cut = mom, cut
        self.regime = nn.Embedding(3, emb)
        self.net = nn.Sequential(nn.Conv1d(1 + cdim + emb, ch // 2, 3, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv1d(ch // 2, ch, 3, padding=1), nn.LeakyReLU(.2), nn.Flatten(), nn.Linear(ch * horizon, 1))

    def forward(self, x, c):
        cond = torch.cat([c, self.regime(_regime(c, self.mom, self.cut))], 1)
        return self.net(torch.cat([x[:, None, :], cond[:, :, None].expand(-1, -1, x.shape[1])], 1)).squeeze(1)


def dcgan_train(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_scale')
    if 'momentum' not in st['cond_names']:
        raise ValueError('La cDCGAN define el régimen con el momentum: activa «use_momentum» en «02 · Condición causal».')
    seed, dev = int(p.get('seed', 42)), _device(p)
    torch.manual_seed(seed)
    zdim, emb, ch, cut = int(p.get('latent_dim', 8)), int(p.get('embedding_dim', 3)), int(p.get('channels', 32)), float(p.get('regime_cut', .5))
    h, cd, mom = st['GX'].shape[1], st['GC'].shape[1], st['cond_names'].index('momentum')
    gen = DCGenerator(h, zdim, cd, emb, mom, cut, ch, bool(p.get('batchnorm', True))).to(dev)
    disc = DCDiscriminator(h, cd, emb, mom, cut, ch).to(dev)
    og = torch.optim.Adam(gen.parameters(), lr=float(p.get('lr_g', 2e-4)), betas=(.5, .999))
    od = torch.optim.Adam(disc.parameters(), lr=float(p.get('lr_d', 2e-4)), betas=(.5, .999))
    tr = st['group'] == 0
    x, c = torch.tensor(st['GX'][tr], device=dev), torch.tensor(st['GC'][tr], device=dev)
    epochs, bs, clip = int(p.get('epochs', 60)), int(p.get('batch_size', 128)), float(p.get('grad_clip', 1.))
    every = max(int(p.get('snapshot_every', 1)), 1)
    g, hist, snaps, t0 = torch.Generator().manual_seed(seed), [], [], time.perf_counter()
    bce = F.binary_cross_entropy_with_logits
    for ep in range(epochs):
        gen.train(); disc.train()
        dl_s = gl_s = 0.
        for idx in _batches(len(x), bs, g):
            if len(idx) < 2:
                continue
            xb, cb = x[idx.to(dev)], c[idx.to(dev)]
            fake = gen(torch.randn(len(xb), zdim, device=dev), cb)
            lr_, lf_ = disc(xb, cb), disc(fake.detach(), cb)                        # paso D (G fijo)
            dl = bce(lr_, torch.ones_like(lr_)) + bce(lf_, torch.zeros_like(lf_))
            _finite('pérdida D', dl)
            od.zero_grad(set_to_none=True); dl.backward(); nn.utils.clip_grad_norm_(disc.parameters(), clip); od.step()
            logits = disc(gen(torch.randn(len(xb), zdim, device=dev), cb), cb)       # paso G (D fijo)
            gl = bce(logits, torch.ones_like(logits))
            _finite('pérdida G', gl)
            og.zero_grad(set_to_none=True); gl.backward(); nn.utils.clip_grad_norm_(gen.parameters(), clip); og.step()
            dl_s += float(dl.detach()) * len(idx); gl_s += float(gl.detach()) * len(idx)
        hist.append({'epoch': ep + 1, 'D': dl_s / len(x), 'G': gl_s / len(x)})
        if (ep + 1) % every == 0 or ep == epochs - 1:
            snaps.append((ep + 1, _snap(gen)))
        if progress:
            progress({'epoch': ep + 1, 'epochs': epochs, 'train': hist[-1]['G'], 'detail': 'cDCGAN'})
    gen = gen.cpu().eval()
    models = {**st.get('models', {}), 'cDCGAN': {'net': gen, 'snaps': snaps, 'kind': 'dcgan'}}
    st = {**st, 'models': models, 'zdim': st.get('zdim', zdim)}
    counts = np.bincount(_regime(torch.tensor(st['GC'][tr]), mom, cut).numpy(), minlength=3)
    return _emit(frame, st, 'gen_dcgan_train'), _report(
        'gen_dcgan_train', selected=f'cDCGAN 1D · z {zdim} · emb régimen {emb} · {ch} canales', splits=_splits(st),
        metrics={'D_final': hist[-1]['D'], 'G_final': hist[-1]['G'], 'segundos': time.perf_counter() - t0, 'instantaneas': len(snaps),
                 **{f'ventanas TR {n.lower()}': int(k) for n, k in zip(REGIMES, counts)}},
        table=[{'bloque': 'Generador', 'forma': f'[z {zdim} | c {cd} | emb {emb}] → Dense {ch}×{h // 4} → ConvT1d {ch}→{ch // 2} → ConvT1d {ch // 2}→1 = {h}'},
               {'bloque': 'Discriminador', 'forma': f'[ventana | c | emb] ({1 + cd + emb} canales × {h}) → Conv1d → Conv1d → logit'}],
        charts=[_chart('Pérdidas alternadas D / G (TR)', hist, ['D', 'G'], 'epoch')],
        notes=['Régimen = embedding aprendido del momentum causal escalado: bajista < −corte < lateral < corte < alcista.',
               'Paso D con G fijo y paso G con D fijo; BCE con logits y optimizadores independientes.',
               'Ventanas de un solo activo: la ampliación multiactivo (embedding de activo) queda pendiente en el pipeline modular.',
               'La selección se hace en «09 · Seleccionar» con ruido fijo en VA.'])


def regime_compare(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None):
    st = _state(frame, 'gen_select')
    if 'momentum' not in st['cond_names']:
        raise ValueError('Comparar regímenes necesita la condición de momentum.')
    sel, sc = st['selected'], st['scaler']
    net = st['chosen'][sel]['net']
    j = st['cond_names'].index('momentum')
    va = np.flatnonzero(st['group'] == 1)
    n = int(p.get('scenarios', 500))
    base = torch.tensor(st['GC'][va[:1]]).repeat(n, 1)
    z = torch.randn(n, net.zdim, generator=torch.Generator().manual_seed(int(p.get('noise_seed', 43))))
    rows, table = [{'sesión': t + 1} for t in range(st['GX'].shape[1])], []
    for name, value in zip(REGIMES, (-1.5, 0., 1.5)):
        cc = base.clone()
        cc[:, j] = value
        with torch.no_grad():
            s = net.generate(z, cc).numpy() * sc['return_std'] + sc['return_mean']
        cum = s.sum(1)
        table.append({'régimen': name, 'momentum escalado': value, 'media acumulada': float(cum.mean()), 'σ acumulada': float(cum.std()),
                      'VaR 5 % acumulado': float(-np.quantile(cum, .05)), 'diversidad (σ entre escenarios)': float(s.std(0).mean())})
        for r, v in zip(rows, np.cumsum(s.mean(0))):
            r[name] = float(v)
    return _emit(frame, st, 'gen_regime_compare'), _report(
        'gen_regime_compare', selected=f'{sel} · mismo ruido, tres regímenes', splits=_splits(st), table=table,
        charts=[_chart('Mismo ruido z · trayectoria media acumulada por régimen', rows, list(REGIMES), 'sesión')],
        notes=['Sólo cambia el momentum de la condición (otros componentes = primera ventana de VA): la diferencia entre curvas es efecto de la condición.',
               'Escenarios hipotéticos, sin cobertura calibrada. No generan órdenes de trading.'])


STEPS = {'gen_windows': windows, 'gen_condition': condition, 'gen_split': split, 'gen_scale': scale, 'gen_cvae': cvae,
         'gen_reparam': reparam, 'gen_vae_train': vae_train, 'gen_gan_train': gan_train, 'gen_select': select,
         'gen_evaluate': evaluate, 'gen_dcgan_train': dcgan_train, 'gen_regime_compare': regime_compare}
