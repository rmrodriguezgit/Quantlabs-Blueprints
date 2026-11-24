"""dl_sequence_classifier: label the next step of a sequence (down/up or down/flat/up)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ....core.contracts import BaseDLTool, Capabilities, Port, RunContext, ToolSpec
from ....core.layers import describe
from ....core.registry import register
from ....core.weights import fit, load_config, seed_all, try_checkpoint
from .model import SequenceClassifier
from .schema import SequenceInput, SequenceParams

CFG = load_config(Path(__file__).parent)


@register
class SequenceClassifierTool(BaseDLTool[SequenceInput, SequenceParams]):
    Input = SequenceInput
    Params = SequenceParams
    spec = ToolSpec(
        name=CFG['name'], version=CFG['version'], title='DL · Clasificador de secuencias',
        description='Capa de entrada GRU → capas intermedias configurables (la última es el embedding) → capa de salida de k clases.',
        inputs=(Port(name='series', type='vector'), Port(name='labels', type='vector', required=False)),
        outputs=(Port(name='label', type='label'), Port(name='probabilities', type='probabilities'),
                 Port(name='latent', type='vector'), Port(name='history_labels', type='vector', description='Predicción por ventana'),
                 Port(name='accuracy', type='scalar', description='Exactitud en holdout')),
        params_schema=SequenceParams.model_json_schema(),
        capabilities=Capabilities(est_ram_mb=CFG['est_ram_mb'], est_vram_mb=CFG['est_vram_mb'],
                                  expected_latency_ms=CFG['expected_latency_ms'], weights_version=CFG['weights_version']),
    )

    def _build(self, context: RunContext) -> tuple[Any, Literal['checkpoint', 'fitted_on_input', 'mock_init']]:
        seed_all(context.seed)
        p = self.params
        model = SequenceClassifier(p.input_units, p.stack(), p.classes)
        hidden = '-'.join(f'{l.units}{l.activation[0]}' for l in p.stack()) or 'none'
        ok = try_checkpoint(model, f'{self.spec.name}-gru{p.input_units}-{hidden}-{p.classes}', CFG['weights_version'])
        return model.to(self.device).eval(), 'checkpoint' if ok else 'mock_init'

    def _warmup(self) -> None:
        with torch.no_grad():
            self.model(torch.zeros(2, self.params.lookback, device=self.device))

    def _targets(self, data: SequenceInput) -> np.ndarray:
        s = np.asarray(data.series, dtype=np.float32)
        if data.labels is not None:
            return np.asarray(data.labels, dtype=np.int64)
        nxt = np.append(s[1:], np.nan)  # label at t = sign of the value at t+1 (the series is a change/return)
        if self.params.classes == 2:
            return np.where(nxt > 0, 1, 0).astype(np.int64)
        dz = self.params.dead_zone
        return np.where(nxt > dz, 2, np.where(nxt < -dz, 0, 1)).astype(np.int64)

    def _infer(self, data: SequenceInput, context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]:
        p = self.params
        raw = np.asarray(data.series, dtype=np.float32)
        y_all = self._targets(data)
        if y_all.max(initial=0) >= p.classes or y_all.min(initial=0) < 0:
            raise ValueError(f'Las etiquetas deben estar en 0…{p.classes - 1}.')
        n = len(raw) - p.lookback + 1
        if n < 20:
            raise ValueError('Serie demasiado corta para el lookback elegido.')
        # window ending at t → label of t. The last window has no known label: it is the live prediction.
        cut = max(1, int((n - 1) * (1 - p.holdout)))
        fit_end = cut + p.lookback - 1
        mu, sd = (float(raw[:fit_end].mean()), float(raw[:fit_end].std()) or 1.0) if p.standardize else (0.0, 1.0)
        s = (raw - mu) / sd
        x = np.stack([s[i:i + p.lookback] for i in range(n)])
        y = y_all[p.lookback - 1:]
        xk, yk = x[:-1], y[:-1]
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        counts = np.bincount(yk[:cut], minlength=p.classes).astype(np.float32)
        weight = t(len(yk[:cut]) / (p.classes * np.maximum(counts, 1))) if p.class_weighting else None
        loss = lambda logits, target: torch.nn.functional.cross_entropy(logits, target, weight=weight)  # noqa: E731
        training: dict[str, Any] = {}
        if self.weights == 'mock_init' and p.fit_on_input and p.epochs:
            training = fit(self.model, t(xk[:cut]), t(yk[:cut]), loss, p, seed=context.seed)
            self.weights = 'fitted_on_input'
        with torch.no_grad():
            proba = torch.softmax(self.model(t(x)), -1).cpu().numpy()
            latent = self.model.encode(t(x[-1:])).cpu().numpy()[0]
        names = CFG['class_names'][str(p.classes)]
        pred = proba.argmax(1)
        ho_true, ho_pred = yk[cut:], pred[cut:-1]
        acc = float(np.mean(ho_pred == ho_true))
        majority = float(np.mean(ho_true == counts.argmax()))
        recall = {names[k]: float(np.mean(ho_pred[ho_true == k] == k)) if (ho_true == k).any() else None for k in range(p.classes)}
        predicted_share = {names[k]: float(np.mean(ho_pred == k)) for k in range(p.classes)}
        last = proba[-1]
        top = int(last.argmax())
        label = names[top] if last[top] >= p.threshold else 'abstain'
        outputs = {'label': label, 'probabilities': {k: float(v) for k, v in zip(names, last)}, 'latent': latent.tolist(),
                   'history_labels': pred.astype(float).tolist(), 'accuracy': acc}
        diag = {'classes': names, 'fit_windows': cut, 'holdout_windows': len(ho_true), 'holdout_accuracy': acc,
                'majority_baseline': majority, 'recall': recall, 'predicted_share': predicted_share, 'threshold': p.threshold,
                'layers': describe(self.model), 'training': training, 'parameters': sum(v.numel() for v in self.model.parameters())}
        return outputs, diag
