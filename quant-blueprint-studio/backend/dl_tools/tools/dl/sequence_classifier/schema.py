"""I/O contract of dl_sequence_classifier."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....core.layers import StackParams
from ....core.optim import OptimParams


class SequenceInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    series: list[float] = Field(min_length=16, max_length=200_000, description='Señal normalizada (p. ej. rendimientos).')
    labels: list[int] | None = Field(None, description='Etiqueta opcional por punto; si falta se usa la dirección del paso siguiente.')

    @model_validator(mode='after')
    def aligned(self) -> 'SequenceInput':
        if any(v != v for v in self.series):
            raise ValueError('La serie contiene NaN.')
        if self.labels is not None and len(self.labels) != len(self.series):
            raise ValueError('labels debe tener la misma longitud que series.')
        return self


class SequenceParams(StackParams, OptimParams):
    model_config = ConfigDict(extra='forbid')
    lookback: int = Field(20, ge=2, le=512)
    input_units: int = Field(32, ge=1, le=2048, description='Celdas de la capa de entrada GRU.')
    classes: int = Field(2, ge=2, le=3, description='Neuronas de la capa de salida: 2 = baja/sube; 3 = baja/plano/sube.')
    dead_zone: float = Field(0.002, ge=0, description='|valor siguiente| ≤ dead_zone → clase plana (sólo con 3 clases).')
    class_weighting: bool = Field(True, description='Pondera la pérdida por la frecuencia de cada clase en el ajuste.')
    holdout: float = Field(0.2, ge=0.05, le=0.5)
    threshold: float = Field(0.5, ge=0.05, le=0.95, description='Probabilidad mínima para emitir la etiqueta; si no, «abstain».')
    standardize: bool = True
    hidden_layers: int = Field(1, ge=0, le=8)
    units: int = Field(16, ge=1, le=2048)
    epochs: int = Field(60, ge=0, le=1000)
    fit_on_input: bool = True
