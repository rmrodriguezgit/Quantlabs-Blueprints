"""I/O contract of dl_recurrent_compare."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....core.layers import StackParams
from ....core.optim import OptimParams


class RecurrentInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: list[list[float]] = Field(min_length=60, description='Indicadores por sesión (n × F), en orden temporal.')
    target: list[float] = Field(description='Objetivo alineado: valor a estimar para la sesión siguiente (NaN en la última).')

    @model_validator(mode='after')
    def aligned(self) -> 'RecurrentInput':
        width = len(self.features[0])
        if width < 1 or any(len(r) != width for r in self.features):
            raise ValueError('features debe ser rectangular.')
        if len(self.target) != len(self.features):
            raise ValueError('target debe tener una fila por sesión.')
        if any(v != v for r in self.features for v in r):
            raise ValueError('features contiene NaN: limpia los datos antes.')
        return self


class RecurrentParams(StackParams, OptimParams):
    model_config = ConfigDict(extra='forbid')
    lookback: int = Field(20, ge=2, le=256, description='Sesiones por secuencia (batch × lookback × F).')
    hidden_size: int = Field(32, ge=2, le=1024, description='Estado oculto de LSTM/GRU.')
    rnn_layers: int = Field(1, ge=1, le=4)
    candidates: list[str] = Field(default_factory=lambda: ['LSTM', 'GRU', 'LSTM (bi)'], description='Modelos a comparar.')
    ridge_alpha: float = Field(1.0, gt=0, le=1e6, description='Ridge sobre ventanas aplanadas (línea base).')
    train_ratio: float = Field(0.7, ge=0.3, le=0.9)
    val_ratio: float = Field(0.15, ge=0.05, le=0.4)
    target_scale: float = Field(10_000.0, gt=0, description='Unidades de salida (10 000 = puntos básicos).')
    hidden_layers: int = Field(0, ge=0, le=8, description='Capas intermedias entre el resumen y la cabeza (0 = como la infografía).')
    epochs: int = Field(60, ge=0, le=1000)
    lr: float = Field(1e-3, gt=0, le=1)
    patience: int = Field(10, ge=1, le=200)

    @model_validator(mode='after')
    def known(self) -> 'RecurrentParams':
        allowed = {'LSTM', 'GRU', 'LSTM (bi)', 'GRU (bi)'}
        bad = [c for c in self.candidates if c not in allowed]
        if bad or not self.candidates:
            raise ValueError(f'Candidatos válidos: {", ".join(sorted(allowed))}.')
        if self.train_ratio + self.val_ratio >= 0.95:
            raise ValueError('TR + VA debe ser < 0.95 para dejar TE.')
        return self
