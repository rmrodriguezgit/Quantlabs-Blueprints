"""Forecaster = one input layer (MLP / 1D-CNN / LSTM) → configurable hidden stack → one Linear output layer."""
from __future__ import annotations

import torch
from torch import nn

from ....core.layers import LayerSpec, activation, hidden_stack


class MLPInput(nn.Module):
    """Input layer: Linear(lookback → width) + activation."""

    def __init__(self, lookback: int, width: int, act: str) -> None:
        super().__init__()
        self.linear, self.act = nn.Linear(lookback, width), activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class CNNInput(nn.Module):
    """Input layer: Conv1d(1 → width, k=3) + activation + global average pooling."""

    def __init__(self, lookback: int, width: int, act: str) -> None:
        super().__init__()
        self.conv, self.act, self.pool = nn.Conv1d(1, width, 3, padding=1), activation(act), nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.conv(x.unsqueeze(1)))).squeeze(-1)


class LSTMInput(nn.Module):
    """Input layer: LSTM(1 → width); the last hidden state summarises the window."""

    def __init__(self, lookback: int, width: int, act: str) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, width, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return out[:, -1]


INPUTS: dict[str, type[nn.Module]] = {'mlp': MLPInput, 'cnn1d': CNNInput, 'lstm': LSTMInput}


class Forecaster(nn.Module):
    def __init__(self, architecture: str, lookback: int, horizon: int, input_units: int, input_activation: str, hidden: list[LayerSpec]) -> None:
        super().__init__()
        self.input_layer = INPUTS[architecture](lookback, input_units, input_activation)
        self.hidden, width = hidden_stack(input_units, hidden)
        self.output_layer = nn.Linear(width, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, lookback) → (B, horizon)
        return self.output_layer(self.hidden(self.input_layer(x)))
