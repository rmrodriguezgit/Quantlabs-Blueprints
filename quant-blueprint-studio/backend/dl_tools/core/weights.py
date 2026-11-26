"""Weights policy shared by tools: checkpoint → fit on the input's own history → seeded mock init.

Missing checkpoints are not errors: the tool logs it and falls back in a controlled, reproducible way.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .optim import OptimParams, make as make_optimizer

log = logging.getLogger('dl_tools')
CHECKPOINTS = Path(__file__).resolve().parents[1] / 'checkpoints'


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(tool_dir: Path) -> dict[str, Any]:
    return json.loads((tool_dir / 'config.json').read_text())


def try_checkpoint(model: torch.nn.Module, name: str, version: str) -> bool:
    """Load checkpoints/<name>-<version>.pt if present and shape-compatible; never raises."""
    path = CHECKPOINTS / f'{name}-{version}.pt'
    if not path.exists():
        log.warning('%s: no checkpoint at %s; using fallback weights.', name, path.name)
        return False
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        return True
    except (RuntimeError, KeyError, OSError) as exc:
        log.warning('%s: checkpoint %s incompatible (%s); using fallback weights.', name, path.name, exc)
        return False


def save_checkpoint(model: torch.nn.Module, name: str, version: str) -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f'{name}-{version}.pt'
    torch.save(model.state_dict(), path)
    return path


def fit(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cfg: 'OptimParams', seed: int = 42, val_fraction: float = 0.15) -> dict[str, Any]:
    """Seeded mini-batch loop with the configured optimizer.

    The last `val_fraction` of the fitting rows (chronological) is an internal validation set: the weights of
    the best validation epoch are restored and training stops after `patience` epochs without improvement.
    """
    seed_all(seed)
    n_val = int(len(x) * val_fraction) if len(x) >= 40 else 0
    xt, yt, xv, yv = x[:len(x) - n_val], y[:len(y) - n_val], x[len(x) - n_val:], y[len(y) - n_val:]
    opt = make_optimizer(model.parameters(), cfg)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    history: list[dict[str, float]] = []
    best, best_state, stale = float('inf'), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(xt), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if len(idx) < 2:          # BatchNorm cannot train on a single row
                continue
            idx = idx.to(x.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                raise ValueError(f'Pérdida no finita en la época {epoch + 1}: baja el learning rate o cambia el optimizador.')
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        row = {'epoch': epoch + 1, 'train': total / max(len(xt), 1)}
        if n_val:
            model.eval()
            with torch.no_grad():
                row['validation'] = float(loss_fn(model(xv), yv))
            if row['validation'] < best - 1e-9:
                best, stale = row['validation'], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if n_val and stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {'history': history, 'epochs_run': len(history), 'best_epoch': (min(history, key=lambda r: r.get('validation', 1e18))['epoch'] if n_val and history else len(history)),
            'early_stopped': len(history) < cfg.epochs, 'val_rows': n_val, 'optimizer': cfg.optimizer}


def windows(series: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Supervised pairs X[t] = s[t-lookback:t], Y[t] = s[t:t+horizon] (past only on the input side)."""
    n = len(series) - lookback - horizon + 1
    if n < 1:
        raise ValueError(f'Serie demasiado corta: {len(series)} puntos para lookback {lookback} + horizonte {horizon}.')
    x = np.stack([series[i:i + lookback] for i in range(n)])
    y = np.stack([series[i + lookback:i + lookback + horizon] for i in range(n)])
    return x, y
