"""Contrato de estado compartido por las herramientas atómicas de las Unidades 2–7 y 10.

No es una herramienta: es el «cable» del pin ``frame``. Cada herramienta recibe un DataFrame,
lee su estado con :func:`state`, devuelve un DataFrame nuevo con :func:`emit` y un informe con
:func:`report`. Ninguna herramienta importa a otra: sólo se comunican a través de este contrato y
del orden del grafo del Blueprint.

Los arrays y redes no caben en ``frame.attrs`` (pandas los copia en profundidad), así que se guardan
en un registro en proceso indexado por id. Cada paso crea una entrada nueva que comparte los objetos
inmutables de la anterior: dos ramas que parten del mismo nodo no se pisan.
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

Progress = Callable[[dict[str, Any]], None] | None
_STORE: 'OrderedDict[str, dict[str, Any]]' = OrderedDict()
_MAX = 48


def state(frame: pd.DataFrame, family: str, need: str | None, labels: dict[str, str]) -> dict[str, Any]:
    """Estado de la familia (``sup`` o ``rl``); exige que ``need`` se haya ejecutado aguas arriba."""
    sid = (frame.attrs.get(family) or {}).get('id')
    st = _STORE.get(sid or '')
    if st is None:
        if need:
            raise ValueError(f'Falta «{labels[need]}» aguas arriba (o vuelve a ejecutar el Blueprint completo).')
        return {}
    if need and need not in st['done']:
        raise ValueError(f'Falta «{labels[need]}» aguas arriba: conecta las herramientas en orden.')
    return st


def emit(frame: pd.DataFrame, family: str, st: dict[str, Any], step: str, out: pd.DataFrame | None = None) -> pd.DataFrame:
    st = {**st, 'done': [*st.get('done', []), step]}
    sid = uuid.uuid4().hex[:12]
    _STORE[sid] = st
    while len(_STORE) > _MAX:
        _STORE.popitem(last=False)
    out = frame.copy() if out is None else out
    out.attrs = {**{k: v for k, v in frame.attrs.items() if k != family}, 'raw_rows': frame.attrs.get('raw_rows', len(frame)),
                 family: {'id': sid, 'steps': st['done']}}
    return out


def report(topic: str, **body: Any) -> dict[str, Any]:
    return {'study': True, 'topic': topic, 'selected': body.pop('selected', ''), 'features': body.pop('features', []),
            'splits': body.pop('splits', {'train': 0, 'validation': 0, 'test': 0}), 'metrics': body.pop('metrics', {}),
            'table': body.pop('table', []), 'charts': body.pop('charts', []), 'notes': body.pop('notes', []), **body}


def chart(title: str, rows: list[dict[str, Any]], series: list[str], x_key: str, chart_type: str = 'line') -> dict[str, Any]:
    return {'title': title, 'rows': rows, 'series': series, 'x_key': x_key, 'chart_type': chart_type}


def splits(st: dict[str, Any]) -> dict[str, int]:
    g = st.get('group')
    if g is None:
        return {'train': 0, 'validation': 0, 'test': 0}
    return {'train': int((g == 0).sum()), 'validation': int((g == 1).sum()), 'test': int((g == 2).sum())}


def names(value: Any) -> list[str]:
    return [s.strip() for s in str(value or '').split(',') if s.strip()]


def ints(value: Any, lo: int, hi: int, what: str) -> list[int]:
    try:
        out = [int(v) for v in names(value)]
    except ValueError as exc:
        raise ValueError(f'{what}: escribe enteros separados por coma.') from exc
    if not out or any(not lo <= v <= hi for v in out):
        raise ValueError(f'{what}: cada valor debe estar entre {lo} y {hi}.')
    return out


def device(p: dict[str, Any]) -> torch.device:
    want = str(p.get('device', 'cpu'))
    if want == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if want == 'cuda' and not torch.cuda.is_available():
        raise ValueError('device = cuda pero no hay GPU disponible: usa cpu o auto.')
    return torch.device(want)


def finite(name: str, value: torch.Tensor | float) -> None:
    t = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not torch.isfinite(t).all():
        raise ValueError(f'Control numérico: {name} contiene NaN/inf. Baja el learning rate o revisa el escalado.')


def day(v: Any) -> str:
    return v.isoformat()[:10] if hasattr(v, 'isoformat') else str(v)


def thin(n: int, limit: int = 400) -> np.ndarray:
    return np.arange(0, n, max(n // limit, 1))
