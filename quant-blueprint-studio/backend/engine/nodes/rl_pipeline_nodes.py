"""Unidad 10 · Double DQN: herramientas atómicas derivadas de la infografía (diez segmentos).

Reutilizan los pasos de datos del catálogo supervisado (``sup_target`` → ``sup_split`` → ``sup_scale``:
segmento 01) y añaden:

==============  ==============================================================================
rl_env          02–03 estado = rasgos + posición, acciones {-1, 0, 1}, recompensa neta con costes
rl_qnet         04    red online y red target (copia)
rl_replay       05–06 exploración ε-greedy y replay buffer con muestreo aleatorio
rl_target       07    objetivo Double DQN (online elige, target evalúa) y γ
rl_sync         09    sincronización online → target y criterio de checkpoint
rl_train        08    bucle de actualización online (Huber, clipping) con lo configurado arriba
rl_baselines    10    referencias: Q-learning tabular, siempre largo, efectivo
rl_evaluate     10    evaluación greedy en test sin aprender
==============  ==============================================================================

Configuración y entrenamiento están separados: rl_qnet / rl_replay / rl_target / rl_sync sólo declaran
su pieza; rl_train ejecuta el bucle con esas piezas. Replay contiene sólo transiciones de TR; validación
elige el checkpoint; test se mide una vez.
"""
from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from . import pipeline_state as S

FAMILY = 'sup'  # comparte el estado de datos de sup_target / sup_split / sup_scale
LABEL = {'sup_scale': 'Escalar sólo con train', 'rl_env': 'Construir el entorno', 'rl_qnet': 'Crear las dos redes',
         'rl_replay': 'Explorar y recordar', 'rl_target': 'Objetivo Double DQN', 'rl_sync': 'Sincronizar y seleccionar',
         'rl_train': 'Actualizar online', 'rl_baselines': 'Referencias', 'rl_evaluate': 'Evaluar sin aprender'}
ACTIONS = np.array([-1, 0, 1])


def _state(frame: pd.DataFrame, need: str) -> dict[str, Any]:
    return S.state(frame, FAMILY, need, LABEL)


def _emit(frame: pd.DataFrame, st: dict[str, Any], step: str) -> pd.DataFrame:
    return S.emit(frame, FAMILY, st, step)


def _step_return(env: dict[str, Any], t: int, prev: int, action: int, terminal: bool) -> float:
    """Posición anterior gana el gap nocturno; la nueva, el intradía; costes por unidad de cambio."""
    gross = (1 + prev * np.expm1(env['gap'][t])) * (1 + action * np.expm1(env['intraday'][t])) - 1
    return float(gross - env['cost'] * abs(action - prev) - (env['cost'] * abs(action) if terminal else 0))


def _rollout(env: dict[str, Any], start: int, end: int, policy: Callable[[np.ndarray], int]) -> tuple[np.ndarray, np.ndarray]:
    prev, rets, sig = 0, [], []
    for t in range(start, end):
        a = int(ACTIONS[policy(np.r_[env['x'][t], prev].astype(np.float32))])
        rets.append(_step_return(env, t, prev, a, t == end - 1))
        sig.append(a)
        prev = a
    return np.array(rets), np.array(sig)


def _score(daily: np.ndarray, signals: np.ndarray) -> dict[str, float]:
    eq = np.cumprod(1 + daily)
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.
    return {'ROI neto': float(eq[-1] - 1), 'Sharpe': float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.,
            'Máx. drawdown': float(np.min(eq / np.maximum.accumulate(np.r_[1., eq])[1:] - 1)),
            'Rotación': float(np.abs(np.diff(np.r_[0, signals, 0])).sum())}


# ── 02–03 · Estado, acciones y entorno ────────────────────────────────────────────────────────────
def env(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'sup_scale')
    g = st['group']
    keep = g >= 0
    idx = np.flatnonzero(keep)
    if np.any(np.diff(g[idx]) < 0):
        raise ValueError('Los bloques deben ser cronológicos TR → VA → TE.')
    cost = float(p.get('cost_bps', 5)) / 1e4
    if not 0 <= cost <= .01:
        raise ValueError('Costes 0–100 bps por unidad de cambio de posición.')
    e = {'x': st['X'][idx], 'gap': st['gap'][idx], 'intraday': st['intraday'][idx], 'dates': st['dates'][idx], 'cost': cost,
         'reward_scale': float(p.get('reward_scale', 100)), 'bounds': (int((g[idx] == 0).sum()), int((g[idx] <= 1).sum()), len(idx))}
    st = {**st, 'env': e}
    a, b, n = e['bounds']
    return _emit(frame, st, 'rl_env'), S.report(
        'rl_env', selected=f'{e["x"].shape[1]} rasgos + posición = {e["x"].shape[1] + 1} entradas · 3 acciones', splits={'train': a, 'validation': b - a, 'test': n - b},
        table=[{'índice': i, 'acción': int(v), 'significado': s} for i, (v, s) in enumerate(zip(ACTIONS, ('corto', 'efectivo', 'largo')))],
        metrics={'costes_bps': cost * 1e4, 'escala_recompensa': e['reward_scale']},
        notes=['s = np.r_[features, position] · actions = [-1, 0, 1].',
               'Decisión al cierre t: la posición anterior participa en el gap nocturno y la nueva en el intradía t+1.',
               f'Recompensa = {e["reward_scale"]:g} × retorno neto (gap + intradía − {cost * 1e4:g} bps por unidad de cambio).'])


# ── 04 · Red online y red target ──────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, d: int, widths: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for w in widths:
            layers += [nn.Linear(d, w), nn.ReLU()]
            d = w
        self.net = nn.Sequential(*layers, nn.Linear(d, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def qnet(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_env')
    widths = S.ints(p.get('widths', '64,32'), 2, 1024, 'Anchos de la Q-network')
    d = st['env']['x'].shape[1] + 1
    st = {**st, 'qnet': {'widths': widths, 'd': d, 'lr': float(p.get('lr', .001)), 'seed': int(p.get('seed', 42))}}
    params = sum(v.numel() for v in QNetwork(d, widths).parameters())
    return _emit(frame, st, 'rl_qnet'), S.report(
        'rl_qnet', selected=f'{d} → {" → ".join(map(str, widths))} → 3 valores Q', splits=S.splits(st), metrics={'parámetros': params},
        notes=['online = QNetwork() · target = copy.deepcopy(online): la online se actualiza, la target sólo se copia.',
               'Q son puntuaciones de cada acción, no probabilidades.'])


# ── 05–06 · Explorar y recordar ───────────────────────────────────────────────────────────────────
def replay(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_qnet')
    cfg = {'capacity': int(p.get('capacity', 10000)), 'warmup': int(p.get('warmup', 128)), 'batch': int(p.get('batch_size', 64)),
           'every': int(p.get('update_every', 4)), 'eps_start': float(p.get('eps_start', 1.)), 'eps_end': float(p.get('eps_end', .05)),
           'eps_steps': int(p.get('eps_decay_steps', 5000))}
    if not cfg['batch'] <= cfg['warmup'] <= cfg['capacity'] or not 0 <= cfg['eps_end'] <= cfg['eps_start'] <= 1:
        raise ValueError('Requiere lote ≤ calentamiento ≤ capacidad y 0 ≤ ε final ≤ ε inicial ≤ 1.')
    st = {**st, 'replay': cfg}
    return _emit(frame, st, 'rl_replay'), S.report(
        'rl_replay', selected=f'buffer {cfg["capacity"]} · lote {cfg["batch"]} · cada {cfg["every"]} pasos', splits=S.splits(st),
        table=[{'parámetro': k, 'valor': v} for k, v in cfg.items()],
        notes=['buffer.append((s, a, r, ns, done)) durante train con acción ε-greedy; ε baja linealmente con los pasos.',
               f'batch = random.sample(buffer, {cfg["batch"]}) cuando hay ≥ {cfg["warmup"]} experiencias. VA y TE nunca entran al buffer.'])


# ── 07 · Objetivo Double DQN ──────────────────────────────────────────────────────────────────────
def target(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_qnet')
    gamma, double = float(p.get('gamma', .95)), bool(p.get('double', True))
    if not 0 <= gamma < 1:
        raise ValueError('γ debe estar en [0, 1).')
    st = {**st, 'bellman': {'gamma': gamma, 'double': double}}
    return _emit(frame, st, 'rl_target'), S.report(
        'rl_target', selected=f'{"Double DQN" if double else "DQN clásica"} · γ = {gamma:g}', splits=S.splits(st),
        notes=['chosen = online(nb).argmax(1) · future = target(nb).gather(1, chosen) · expected = rb + γ·(1 − done)·future.' if double else
               'DQN clásica: future = target(nb).max(1): tiende a sobreestimar Q.', 'Si done = 1 no hay valor futuro.'])


# ── 09 · Sincronizar y seleccionar ────────────────────────────────────────────────────────────────
def sync(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_qnet')
    every = int(p.get('sync_every', 100))
    if not 1 <= every <= 100000:
        raise ValueError('Sincronización cada 1–100000 actualizaciones.')
    st = {**st, 'sync': {'every': every, 'metric': str(p.get('checkpoint_metric', 'ROI neto'))}}
    return _emit(frame, st, 'rl_sync'), S.report(
        'rl_sync', selected=f'copiar cada {every} actualizaciones · checkpoint por {st["sync"]["metric"]} en VA', splits=S.splits(st),
        notes=['target.load_state_dict(online.state_dict()) cada N actualizaciones.', 'Al final de cada época se evalúa la política greedy en VA y se guarda el mejor checkpoint.'])


# ── 08 · Actualizar online (bucle de entrenamiento) ───────────────────────────────────────────────
def train(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_qnet')
    for need in ('rl_replay', 'rl_target', 'rl_sync'):
        _state(frame, need)
    e, q, rp, bl, sy = st['env'], st['qnet'], st['replay'], st['bellman'], st['sync']
    seed = q['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    online = QNetwork(q['d'], q['widths'])
    tgt = deepcopy(online).eval()
    opt = torch.optim.Adam(online.parameters(), lr=q['lr'])
    a, b, _ = e['bounds']
    buf: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    steps = updates = 0
    best, best_state, hist, t0 = -np.inf, deepcopy(online.state_dict()), [], time.perf_counter()
    epochs = int(p.get('epochs', 12))

    def greedy(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(online(torch.tensor(obs)).argmax())

    for ep in range(epochs):
        prev, losses = 0, []
        for t in range(a):
            obs = np.r_[e['x'][t], prev].astype(np.float32)
            eps = max(rp['eps_end'], rp['eps_start'] - (rp['eps_start'] - rp['eps_end']) * steps / max(rp['eps_steps'], 1))
            ai = random.randrange(3) if random.random() < eps else greedy(obs)
            act, done = int(ACTIONS[ai]), t == a - 1
            reward = e['reward_scale'] * _step_return(e, t, prev, act, done)
            nxt = np.r_[e['x'][t + 1] if not done else np.zeros_like(e['x'][t]), act].astype(np.float32)
            buf.append((obs, ai, reward, nxt, done))
            buf = buf[-rp['capacity']:]
            steps, prev = steps + 1, act
            if len(buf) >= rp['warmup'] and steps % rp['every'] == 0:
                batch = random.sample(buf, rp['batch'])
                ob = torch.tensor(np.stack([r[0] for r in batch]))
                ac = torch.tensor([r[1] for r in batch])
                re = torch.tensor([r[2] for r in batch], dtype=torch.float32)
                nb = torch.tensor(np.stack([r[3] for r in batch]))
                db = torch.tensor([r[4] for r in batch], dtype=torch.float32)
                with torch.no_grad():
                    future = tgt(nb).gather(1, online(nb).argmax(1, keepdim=True)).squeeze(1) if bl['double'] else tgt(nb).max(1).values
                    expected = re + bl['gamma'] * (1 - db) * future
                current = online(ob).gather(1, ac[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(current, expected)
                S.finite('pérdida Q', loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), float(p.get('grad_clip', 1.)))
                opt.step()
                updates += 1
                losses.append(float(loss.detach()))
                if updates % sy['every'] == 0:
                    tgt.load_state_dict(online.state_dict())
        online.eval()
        val = _score(*_rollout(e, a, b, greedy))
        score = val.get(sy['metric'], val['ROI neto'])
        hist.append({'epoch': ep + 1, 'validación': score, 'pérdida Q': float(np.mean(losses)) if losses else 0.})
        if progress:
            progress({'epoch': ep + 1, 'epochs': epochs, 'validation': score, 'detail': 'Double DQN · VA'})
        if score > best:
            best, best_state = score, deepcopy(online.state_dict())
        online.train()
    online.load_state_dict(best_state)
    online.eval()
    st = {**st, 'policies': {**st.get('policies', {}), 'Double DQN' if bl['double'] else 'DQN': online}}
    return _emit(frame, st, 'rl_train'), S.report(
        'rl_train', selected=f'{epochs} épocas · {updates} actualizaciones · mejor {sy["metric"]} VA {best:.4f}', splits=S.splits(st),
        metrics={'actualizaciones': updates, 'pasos': steps, 'mejor_VA': float(best), 'segundos': time.perf_counter() - t0},
        charts=[S.chart(f'Checkpoint · {sy["metric"]} en validación', hist, ['validación'], 'epoch'),
                S.chart('Pérdida Huber media por época', hist, ['pérdida Q'], 'epoch')],
        notes=['loss = F.smooth_l1_loss(current, expected) · clipping de norma a 1 · Adam.', 'Validación se recorre en greedy sin actualizar pesos.'])


# ── 10 · Referencias ──────────────────────────────────────────────────────────────────────────────
def baselines(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_env')
    e = st['env']
    a = e['bounds'][0]
    rng = random.Random(int(p.get('seed', 42)))
    pols: dict[str, Any] = {}
    if p.get('q_learning', True):
        q, bins = np.zeros((5, 3, 3)), np.array([-1., -.25, .25, 1.])
        for _ in range(int(p.get('epochs', 12))):
            prev = 0
            for t in range(a):
                s, pi = int(np.digitize(e['x'][t, 0], bins)), prev + 1
                ai = rng.randrange(3) if rng.random() < .1 else int(q[s, pi].argmax())
                act, done = int(ACTIONS[ai]), t == a - 1
                fut = 0 if done else float(q[int(np.digitize(e['x'][t + 1, 0], bins)), act + 1].max())
                q[s, pi, ai] += .1 * (e['reward_scale'] * _step_return(e, t, prev, act, done) + .95 * fut - q[s, pi, ai])
                prev = act
        pols['Q-learning'] = ('table', q, bins)
    if p.get('always_long', True):
        pols['Siempre largo'] = ('const', 2)
    if p.get('cash', True):
        pols['Efectivo'] = ('const', 1)
    st = {**st, 'baselines': pols}
    return _emit(frame, st, 'rl_baselines'), S.report(
        'rl_baselines', selected=', '.join(pols) or 'ninguna', splits=S.splits(st),
        notes=['Q-learning tabular: primer rasgo escalado discretizado en 5 tramos × posición; entrenado sólo en TR.',
               'Siempre largo y efectivo no aprenden: fijan el listón mínimo con los mismos costes.'])


def _policy(spec: Any) -> Callable[[np.ndarray], int]:
    if isinstance(spec, nn.Module):
        return lambda obs: int(spec(torch.tensor(obs)).argmax())
    if spec[0] == 'const':
        return lambda _obs: spec[1]
    _, q, bins = spec
    return lambda obs: int(q[int(np.digitize(obs[0], bins)), int(obs[-1]) + 1].argmax())


def evaluate(frame: pd.DataFrame, p: dict[str, Any], progress: S.Progress = None):
    st = _state(frame, 'rl_train')
    e = st['env']
    _, b, n = e['bounds']
    rows = [{'date': S.day(d)} for d in e['dates'][b:n]]
    table, summary = [], {}
    with torch.no_grad():
        for name, spec in {**st['policies'], **st.get('baselines', {})}.items():
            daily, sig = _rollout(e, b, n, _policy(spec))
            sc = _score(daily, sig)
            table.append({'política': name, **sc})
            if name in st['policies']:
                summary = sc
            for r, v in zip(rows, np.cumprod(1 + daily)):
                r[name] = float(v - 1)
    return _emit(frame, st, 'rl_evaluate'), S.report(
        'rl_evaluate', selected=next(iter(st['policies'])), scope='TE · greedy, sin exploración ni actualizaciones', splits=S.splits(st),
        metrics=summary, table=table, charts=[S.chart('Test greedy · gap + intradía − costes', rows, [t['política'] for t in table], 'date')],
        notes=["rollout(greedy_dqn, 'test'): acción = argmax Q, ε = 0 y pesos congelados.",
               'Simulación educativa: una menor pérdida Q no garantiza mayor rentabilidad.'])


STEPS = {'rl_env': env, 'rl_qnet': qnet, 'rl_replay': replay, 'rl_target': target, 'rl_sync': sync, 'rl_train': train,
         'rl_baselines': baselines, 'rl_evaluate': evaluate}
