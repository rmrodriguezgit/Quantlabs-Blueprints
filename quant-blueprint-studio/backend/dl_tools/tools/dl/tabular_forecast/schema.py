"""I/O contract of dl_tabular_forecast."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ....core.layers import Activation, StackParams
from ....core.optim import OptimParams


class ForecastInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    series: list[float] = Field(min_length=16, max_length=200_000, description='Serie numérica (ya escalada o rendimientos).')

    @field_validator('series')
    @classmethod
    def finite(cls, v: list[float]) -> list[float]:
        if any(x != x or x in (float('inf'), float('-inf')) for x in v):
            raise ValueError('La serie contiene NaN o infinitos.')
        return v


class ForecastParams(StackParams, OptimParams):
    model_config = ConfigDict(extra='forbid')
    architecture: Literal['mlp', 'cnn1d', 'lstm'] = Field('lstm', description='Tipo de la capa de entrada.')
    input_units: int = Field(32, ge=1, le=2048, description='Neuronas / filtros / celdas de la capa de entrada.')
    input_activation: Activation = 'gelu'
    lookback: int = Field(20, ge=2, le=512)
    horizon: int = Field(1, ge=1, le=64, description='Neuronas de la capa de salida (pasos a predecir).')
    holdout: float = Field(0.2, ge=0.05, le=0.5, description='Fracción final de ventanas fuera del ajuste (métricas honestas).')
    coverage: float = Field(0.9, ge=0.5, le=0.99)
    standardize: bool = Field(True, description='Escala la serie con media/desv. del ajuste (y deshace al predecir).')
    epochs: int = Field(60, ge=0, le=1000)
    fit_on_input: bool = True
