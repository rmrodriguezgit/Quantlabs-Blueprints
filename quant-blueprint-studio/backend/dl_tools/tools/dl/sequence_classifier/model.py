"""Classifier = one GRU input layer → configurable hidden stack (its last activation is the latent embedding) → one Linear output layer."""
from __future__ import annotations

import torch
from torch import nn

from ....core.layers import LayerSpec, hidden_stack


class GRUInput(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(1, width, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x.unsqueeze(-1))
        return out[:, -1]


class SequenceClassifier(nn.Module):
    def __init__(self, input_units: int, hidden: list[LayerSpec], classes: int) -> None:
        super().__init__()
        self.input_layer = GRUInput(input_units)
        self.hidden, width = hidden_stack(input_units, hidden)
        self.output_layer = nn.Linear(width, classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, latent): output of the last hidden layer
        return self.hidden(self.input_layer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits (B, classes)
        return self.output_layer(self.encode(x))
