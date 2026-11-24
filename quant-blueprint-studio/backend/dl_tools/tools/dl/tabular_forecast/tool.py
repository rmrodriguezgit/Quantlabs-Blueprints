"""dl_tabular_forecast: multi-step forecast of a numeric series (input layer MLP / 1D-CNN / LSTM)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ....core.contracts import BaseDLTool, Capabilities, Port, RunContext, ToolSpec
from ....core.layers import describe
from ....core.registry import register
from ....core.weights import fit, load_config, seed_all, try_checkpoint, windows
from .model import Forecaster
from .schema import ForecastInput, ForecastParams

CFG = load_config(Path(__file__).parent)


@register
class TabularForecastTool(BaseDLTool[ForecastInput, ForecastParams]):
    Input = ForecastInput
    Params = ForecastParams
    spec = ToolSpec(
        name=CFG['name'], version=CFG['version'], title='DL · Pronóstico de serie',
        description='Capa de entrada (MLP/CNN/LSTM) → capas intermedias configurables → capa de salida de h neuronas.',
        inputs=(Port(name='series', type='vector', description='Serie numérica'),),
        outputs=(Port(name='forecast', type='vector', description='h valores siguientes'),
                 Port(name='lower', type='vector'), Port(name='upper', type='vector'),
                 Port(name='residuals', type='matrix', description='Errores de las ventanas holdout (n × h)'),
                 Port(name='confidence', type='scalar', description='1 − RMSE_modelo/RMSE_línea_base en holdout')),
        params_schema=ForecastParams.model_json_schema(),
        capabilities=Capabilities(est_ram_mb=CFG['est_ram_mb'], est_vram_mb=CFG['est_vram_mb'],
                                  expected_latency_ms=CFG['expected_latency_ms'], weights_version=CFG['weights_version']),
    )

    def _shape_key(self) -> str:
        p = self.params
        hidden = '-'.join(f'{l.units}{l.activation[0]}' for l in p.stack()) or 'none'
        return f'{self.spec.name}-{p.architecture}{p.input_units}-{hidden}-{p.lookback}x{p.horizon}'

    def _build(self, context: RunContext) -> tuple[Any, Literal['checkpoint', 'fitted_on_input', 'mock_init']]:
        seed_all(context.seed)
        p = self.params
        model = Forecaster(p.architecture, p.lookback, p.horizon, p.input_units, p.input_activation, p.stack())
        ok = try_checkpoint(model, self._shape_key(), CFG['weights_version'])
        return model.to(self.device).eval(), 'checkpoint' if ok else 'mock_init'

    def _warmup(self) -> None:
        with torch.no_grad():
            self.model(torch.zeros(2, self.params.lookback, device=self.device))

    def _infer(self, data: ForecastInput, context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]:
        p = self.params
        raw = np.asarray(data.series, dtype=np.float32)
        x_raw, y_raw = windows(raw, p.lookback, p.horizon)
        cut = max(1, int(len(x_raw) * (1 - p.holdout)))
        if cut >= len(x_raw):
            raise ValueError('No quedan ventanas holdout: alarga la serie o reduce lookback/horizonte.')
        # standardise with statistics of the fitting part only (the holdout never leaks into the scale)
        fit_end = cut + p.lookback + p.horizon - 1
        mu, sd = (float(raw[:fit_end].mean()), float(raw[:fit_end].std()) or 1.0) if p.standardize else (0.0, 1.0)
        s = (raw - mu) / sd
        x, y = windows(s, p.lookback, p.horizon)
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        training: dict[str, Any] = {}
        if self.weights == 'mock_init' and p.fit_on_input and p.epochs:
            training = fit(self.model, t(x[:cut]), t(y[:cut]), torch.nn.functional.mse_loss, p, seed=context.seed)
            self.weights = 'fitted_on_input'
        with torch.no_grad():
            pred_ho = self.model(t(x[cut:])).cpu().numpy() * sd + mu
            nxt = self.model(t(s[-p.lookback:][None])).cpu().numpy()[0] * sd + mu
        resid = y_raw[cut:] - pred_ho
        q = np.quantile(np.abs(resid), p.coverage, axis=0)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        last = float(np.sqrt(np.mean((y_raw[cut:] - x_raw[cut:, -1:]) ** 2)))
        mean = float(np.sqrt(np.mean((y_raw[cut:] - y_raw[:cut].mean()) ** 2)))
        naive = min(last, mean) or 1e-12  # the stronger baseline: last value (levels) or the fitted mean (returns)
        hit = float(np.mean(np.sign(pred_ho[:, 0] - (0 if mean <= last else x_raw[cut:, -1])) == np.sign(y_raw[cut:, 0] - (0 if mean <= last else x_raw[cut:, -1]))))
        outputs = {'forecast': nxt.tolist(), 'lower': (nxt - q).tolist(), 'upper': (nxt + q).tolist(),
                   'residuals': resid.tolist(), 'confidence': float(1 - rmse / naive)}
        diag = {'architecture': p.architecture, 'fit_windows': cut, 'holdout_windows': len(x) - cut, 'holdout_rmse': rmse,
                'naive_rmse': naive, 'baseline': 'último valor' if last <= mean else 'media del ajuste', 'direction_hit': hit,
                'coverage_target': p.coverage, 'layers': describe(self.model), 'training': training,
                'parameters': sum(v.numel() for v in self.model.parameters())}
        return outputs, diag
