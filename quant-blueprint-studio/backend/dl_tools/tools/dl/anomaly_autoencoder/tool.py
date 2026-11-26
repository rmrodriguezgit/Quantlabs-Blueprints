"""dl_anomaly_autoencoder: reconstruction-error anomaly score, is_anomaly flag and per-dimension error."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ....core.contracts import BaseDLTool, Capabilities, Port, RunContext, ToolSpec
from ....core.layers import describe
from ....core.registry import register
from ....core.weights import fit, load_config, seed_all, try_checkpoint
from .model import AutoEncoder
from .schema import AnomalyInput, AnomalyParams

CFG = load_config(Path(__file__).parent)


@register
class AnomalyAutoencoderTool(BaseDLTool[AnomalyInput, AnomalyParams]):
    Input = AnomalyInput
    Params = AnomalyParams
    spec = ToolSpec(
        name=CFG['name'], version=CFG['version'], title='DL · Anomalías (autoencoder)',
        description='Reconstruye cada fila; el error de reconstrucción es el score de anomalía frente a un umbral fijado en el ajuste.',
        inputs=(Port(name='data', type='matrix', description='n × d'),),
        outputs=(Port(name='score', type='vector', description='Error por fila'), Port(name='is_anomaly', type='mask'),
                 Port(name='dimension_error', type='matrix', description='Error por fila y dimensión'),
                 Port(name='threshold', type='scalar'), Port(name='anomaly_rate', type='scalar')),
        params_schema=AnomalyParams.model_json_schema(),
        capabilities=Capabilities(est_ram_mb=CFG['est_ram_mb'], est_vram_mb=CFG['est_vram_mb'],
                                  expected_latency_ms=CFG['expected_latency_ms'], weights_version=CFG['weights_version']),
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._width = 0

    def _build(self, context: RunContext) -> tuple[Any, Literal['checkpoint', 'fitted_on_input', 'mock_init']]:
        # the input width is only known at execute time; the model is (re)built lazily in _infer
        return torch.nn.Identity(), 'mock_init'

    def _ensure(self, width: int, context: RunContext) -> None:
        if self._width == width:
            return
        seed_all(context.seed)
        p = self.params
        if p.latent >= width:
            raise ValueError(f'El cuello de botella ({p.latent}) debe ser menor que el número de columnas ({width}); si no, el autoencoder copia la entrada y no detecta nada.')
        model = AutoEncoder(width, p.input_units, p.input_activation, p.stack(), p.latent)
        hidden = '-'.join(f'{l.units}{l.activation[0]}' for l in p.stack()) or 'none'
        ok = try_checkpoint(model, f'{self.spec.name}-{width}-{p.input_units}-{hidden}-z{p.latent}', CFG['weights_version'])
        self.model, self.weights, self._width = model.to(self.device).eval(), ('checkpoint' if ok else 'mock_init'), width

    def unload(self) -> None:
        super().unload()
        self._width = 0

    def _infer(self, data: AnomalyInput, context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]:
        p = self.params
        x = np.asarray(data.data, dtype=np.float32)
        n_fit = max(4, int(len(x) * p.fit_fraction))
        mu, sd = x[:n_fit].mean(0), x[:n_fit].std(0) + 1e-8  # standardise with fit rows only
        z = (x - mu) / sd
        self._ensure(z.shape[1], context)
        t = torch.as_tensor(z, device=self.device)
        loss = torch.nn.functional.mse_loss if p.error == 'mse' else torch.nn.functional.l1_loss
        training: dict[str, Any] = {}
        if self.weights == 'mock_init' and p.fit_on_input and p.epochs:
            training = fit(self.model, t[:n_fit], t[:n_fit], loss, p, seed=context.seed)
            self.weights = 'fitted_on_input'
        with torch.no_grad():
            diff = (self.model(t) - t).cpu().numpy()
        dim = diff ** 2 if p.error == 'mse' else np.abs(diff)
        score = dim.mean(1)
        if p.threshold_mode == 'quantile':
            if not 0 < p.threshold < 1:
                raise ValueError('Con threshold_mode=quantile el umbral debe estar en (0, 1).')
            thr = float(np.quantile(score[:n_fit], p.threshold))
        else:
            thr = float(p.threshold)
        flag = score > thr
        outputs = {'score': score.tolist(), 'is_anomaly': flag.tolist(), 'dimension_error': dim.tolist(),
                   'threshold': thr, 'anomaly_rate': float(flag.mean())}
        diag = {'rows': len(x), 'dims': int(x.shape[1]), 'fit_rows': n_fit, 'anomalies': int(flag.sum()),
                'anomalies_after_fit': int(flag[n_fit:].sum()), 'worst_dimension': int(dim[flag].mean(0).argmax()) if flag.any() else None,
                'fit_error': float(score[:n_fit].mean()), 'after_fit_error': float(score[n_fit:].mean()) if len(score) > n_fit else None,
                'layers': describe(self.model), 'training': training, 'parameters': sum(v.numel() for v in self.model.parameters())}
        return outputs, diag
