"""Tool registry: discovers tools/dl/<name>/tool.py packages and exposes their ToolSpecs as JSON."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .contracts import BaseDLTool, ToolSpec

_TOOLS: dict[str, type[BaseDLTool[Any, Any]]] = {}


def register(cls: type[BaseDLTool[Any, Any]]) -> type[BaseDLTool[Any, Any]]:
    """Class decorator used by each tool module. Names must be unique."""
    name = cls.spec.name
    if name in _TOOLS and _TOOLS[name] is not cls:
        raise ValueError(f'Tool {name} already registered.')
    _TOOLS[name] = cls
    return cls


def discover() -> dict[str, type[BaseDLTool[Any, Any]]]:
    """Import every tools.dl.<pkg>.tool module once; each registers itself."""
    from ..tools import dl
    for module in pkgutil.iter_modules(dl.__path__):
        if module.ispkg:
            importlib.import_module(f'{dl.__name__}.{module.name}.tool')
    return dict(_TOOLS)


def get(name: str) -> type[BaseDLTool[Any, Any]]:
    if name not in _TOOLS:   # a tool imported directly registers alone: discover the rest on demand
        discover()
    if name not in _TOOLS:
        raise KeyError(f'Herramienta desconocida: {name}. Disponibles: {", ".join(sorted(_TOOLS))}')
    return _TOOLS[name]


def specs() -> list[ToolSpec]:
    return [cls.spec for cls in discover().values()]


def describe() -> list[dict[str, Any]]:
    """JSON the Blueprint UI/engine can store next to graphs."""
    return [s.model_dump(mode='json') for s in specs()]
