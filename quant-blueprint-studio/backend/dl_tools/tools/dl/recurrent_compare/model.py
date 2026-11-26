"""Recurrent regressor exactly as in «Cómo programar: LSTM y GRU» (segments 02–07).

Input layer  : nn.LSTM or nn.GRU (optionally bidirectional)            — segments 02, 03, 04
Final state  : LSTM returns (h_n, c_n), GRU returns h_n                 — segment 05
Summary      : cat(h[-2], h[-1]) if bidirectional else h[-1]            — segment 06
Hidden layers: optional configurable stack (0 by default, as in the infographic)
Output layer : nn.Linear(64 if bidirectional else 32, 1) → scalar return — segment 07
No state is passed between windows: every forward call starts a fresh memory (segment 08).
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ....core.layers import LayerSpec, hidden_stack

Kind = Literal['LSTM', 'GRU']


class RecurrentRegressor(nn.Module):
    def __init__(self, kind: Kind, features: int, hidden: int, bidirectional: bool, layers: int = 1,
                 stack: list[LayerSpec] | None = None) -> None:
        super().__init__()
        self.kind, self.bidirectional = kind, bidirectional
        cls = nn.LSTM if kind == 'LSTM' else nn.GRU
        self.input_layer = cls(features, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        width = hidden * (2 if bidirectional else 1)
        self.hidden, width = hidden_stack(width, stack or [])
        self.output_layer = nn.Linear(width, 1)          # self.head in the infographic

    def summary(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.input_layer(x)                   # fresh state per window: nothing carried over
        h = state[0] if self.kind == 'LSTM' else state   # LSTM → (h_n, c_n); GRU → h_n
        return torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, features) → (batch,)
        return self.output_layer(self.hidden(self.summary(x))).squeeze(-1)
