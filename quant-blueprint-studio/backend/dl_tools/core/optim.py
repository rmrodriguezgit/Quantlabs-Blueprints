"""Optimizer choice shared by every tool (configured from the Inspector)."""
from __future__ import annotations

from typing import Iterable, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

OptimizerName = Literal['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad']


class OptimParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    optimizer: OptimizerName = 'adam'
    lr: float = Field(3e-3, gt=0, le=1)
    weight_decay: float = Field(0.0, ge=0, le=1)
    momentum: float = Field(0.9, ge=0, lt=1, description='Sólo SGD / RMSprop.')
    epochs: int = Field(30, ge=0, le=1000)
    batch_size: int = Field(64, ge=4, le=4096)
    patience: int = Field(10, ge=1, le=200, description='Épocas sin mejora en validación interna antes de parar.')


def make(params: Iterable[torch.nn.Parameter], cfg: OptimParams) -> torch.optim.Optimizer:
    if cfg.optimizer == 'adam':
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'rmsprop':
        return torch.optim.RMSprop(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    return torch.optim.Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
