"""I/O contract of dl_anomaly_autoencoder."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....core.layers import Activation, StackParams
from ....core.optim import OptimParams


class AnomalyInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    data: list[list[float]] = Field(min_length=8, max_length=200_000, description='Matriz n × d (filas = observaciones).')

    @model_validator(mode='after')
    def rectangular(self) -> 'AnomalyInput':
        width = len(self.data[0])
        if width < 1 or any(len(r) != width for r in self.data):
            raise ValueError('data debe ser rectangular (todas las filas con igual número de columnas).')
        if any(v != v for r in self.data for v in r):
            raise ValueError('data contiene NaN.')
        return self


class AnomalyParams(StackParams, OptimParams):
    model_config = ConfigDict(extra='forbid')
    hidden_layers: int = Field(1, ge=0, le=8, description='Capas intermedias del codificador (el decodificador las refleja).')
    units: int = Field(16, ge=1, le=2048)
    input_units: int = Field(32, ge=1, le=2048, description='Neuronas de la capa de entrada (y de la última del decodificador).')
    input_activation: Activation = 'gelu'
    latent: int = Field(1, ge=1, le=64, description='Cuello de botella; debe ser menor que el nº de columnas.')
    error: Literal['mse', 'mae'] = 'mse'
    threshold_mode: Literal['quantile', 'absolute'] = 'quantile'
    threshold: float = Field(0.99, gt=0, description='Cuantil del error en el ajuste (0–1) o umbral absoluto.')
    fit_fraction: float = Field(0.7, ge=0.2, le=1.0, description='Filas iniciales usadas para ajustar y fijar el umbral.')
    epochs: int = Field(150, ge=0, le=1000)
    lr: float = Field(5e-3, gt=0, le=1)
    fit_on_input: bool = True
