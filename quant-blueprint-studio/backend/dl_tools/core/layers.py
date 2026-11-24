"""Configurable intermediate layers shared by every tool.

Each tool has exactly ONE input layer (its own: Linear, Conv1d, LSTM, GRU…) and ONE output layer (Linear head).
Between them sits a stack of 0–8 hidden layers described by LayerSpec, so depth, width, activation,
dropout and normalisation are configured without touching the model code.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from torch import nn

Activation = Literal['relu', 'gelu', 'tanh', 'sigmoid', 'swish', 'leaky_relu', 'elu', 'identity']
ACTIVATIONS: dict[str, type[nn.Module]] = {
    'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'swish': nn.SiLU,
    'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU, 'identity': nn.Identity,
}


class LayerSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    units: int = Field(32, ge=1, le=2048)
    activation: Activation = 'gelu'
    dropout: float = Field(0.0, ge=0, lt=1)
    batch_norm: bool = False


class StackParams(BaseModel):
    """Mixin for tool params: either an explicit `layers` list or n identical layers."""
    model_config = ConfigDict(extra='forbid')
    hidden_layers: int = Field(2, ge=0, le=8, description='Número de capas intermedias.')
    units: int = Field(32, ge=1, le=2048, description='Neuronas por capa intermedia.')
    activation: Activation = 'gelu'
    dropout: float = Field(0.0, ge=0, lt=1)
    batch_norm: bool = False
    layers: list[LayerSpec] | None = Field(None, max_length=8, description='Capas intermedias explícitas (sustituye a las cuatro anteriores).')

    def stack(self) -> list[LayerSpec]:
        if self.layers is not None:
            return list(self.layers)
        return [LayerSpec(units=self.units, activation=self.activation, dropout=self.dropout, batch_norm=self.batch_norm)
                for _ in range(self.hidden_layers)]


def activation(name: str) -> nn.Module:
    return ACTIVATIONS[name]()


def hidden_stack(in_features: int, specs: list[LayerSpec]) -> tuple[nn.Sequential, int]:
    """Linear → [BatchNorm] → activation → [Dropout] per spec. Returns the stack and its output width."""
    mods: list[nn.Module] = []
    width = in_features
    for spec in specs:
        mods.append(nn.Linear(width, spec.units))
        if spec.batch_norm:
            mods.append(nn.BatchNorm1d(spec.units))
        mods.append(activation(spec.activation))
        if spec.dropout:
            mods.append(nn.Dropout(spec.dropout))
        width = spec.units
    return nn.Sequential(*mods), width


def parse_layers(text: str) -> list[LayerSpec]:
    """UI shorthand «64:relu:0.1, 32:tanh» → LayerSpec list (units[:activation[:dropout[:bn]]])."""
    out: list[LayerSpec] = []
    for chunk in (c.strip() for c in text.split(',') if c.strip()):
        parts = [p.strip() for p in chunk.split(':')]
        spec: dict[str, object] = {'units': int(parts[0])}
        if len(parts) > 1 and parts[1]:
            spec['activation'] = parts[1].lower()
        if len(parts) > 2 and parts[2]:
            spec['dropout'] = float(parts[2])
        if len(parts) > 3:
            spec['batch_norm'] = parts[3].lower() in ('bn', 'true', '1', 'batchnorm')
        out.append(LayerSpec.model_validate(spec))
    return out


def describe(model: nn.Module) -> list[dict[str, object]]:
    """Layer table for reports: role, module, parameters."""
    rows: list[dict[str, object]] = []
    for role in ('input_layer', 'hidden', 'encoder_hidden', 'bottleneck', 'decoder_hidden', 'output_layer'):
        part = getattr(model, role, None)
        if part is None:
            continue
        mods = list(part) if isinstance(part, nn.Sequential) else [part]
        for m in mods or [nn.Identity()]:
            rows.append({'capa': role, 'módulo': m.__class__.__name__, 'detalle': m.extra_repr()[:60],
                         'parámetros': sum(p.numel() for p in m.parameters())})
    return rows
