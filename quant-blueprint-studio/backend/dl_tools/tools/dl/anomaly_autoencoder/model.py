"""Autoencoder = one input layer → encoder hidden stack → bottleneck → mirrored decoder stack → one output layer."""
from __future__ import annotations

import torch
from torch import nn

from ....core.layers import LayerSpec, activation, hidden_stack


class AutoEncoder(nn.Module):
    def __init__(self, width: int, input_units: int, input_activation: str, hidden: list[LayerSpec], latent: int) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(width, input_units), activation(input_activation))
        self.encoder_hidden, w = hidden_stack(input_units, hidden)
        self.bottleneck = nn.Linear(w, latent)
        # the decoder mirrors the encoder widths in reverse and ends at the input-layer width
        mirrored = [LayerSpec(units=s.units, activation=s.activation, dropout=s.dropout, batch_norm=s.batch_norm) for s in reversed(hidden)]
        mirrored.append(LayerSpec(units=input_units, activation=input_activation))
        self.decoder_hidden, w = hidden_stack(latent, mirrored)
        self.output_layer = nn.Linear(w, width)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.encoder_hidden(self.input_layer(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.decoder_hidden(self.encode(x)))
