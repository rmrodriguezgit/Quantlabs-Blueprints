"""Typed contracts every DL tool implements: ports, capabilities, ToolSpec, structured errors and BaseDLTool.

A tool never imports another tool, the UI or global mutable state. The Blueprint engine only sees the
ToolSpec (to validate and serialise graphs) and calls the lifecycle: load → warmup → execute → unload.
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .device import DeviceName, resolve_device
from .memory import release

log = logging.getLogger('dl_tools')

# Port payload types. Two ports can be connected only when their types are equal (or the input accepts "any").
PortType = Literal['vector', 'matrix', 'scalar', 'label', 'probabilities', 'mask', 'table', 'any']


class Port(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    type: PortType
    description: str = ''
    required: bool = True


class Capabilities(BaseModel):
    model_config = ConfigDict(frozen=True)
    devices: tuple[DeviceName, ...] = ('cpu', 'cuda', 'mps')
    est_ram_mb: float = Field(ge=0)
    est_vram_mb: float = Field(ge=0)
    expected_latency_ms: float = Field(ge=0)
    weights_version: str
    deterministic: bool = True


class ToolSpec(BaseModel):
    """NodeDescriptor: what the Blueprint registry discovers, validates and serialises."""
    model_config = ConfigDict(frozen=True)
    name: str
    version: str
    title: str
    description: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    params_schema: dict[str, Any]
    capabilities: Capabilities

    def input(self, name: str) -> Port:
        return next(p for p in self.inputs if p.name == name)

    def output(self, name: str) -> Port:
        return next(p for p in self.outputs if p.name == name)


class ToolError(Exception):
    """Structured failure: a code the engine can route on, the tool name and machine-readable details."""

    def __init__(self, code: Literal['invalid_input', 'invalid_params', 'not_loaded', 'execution_failed', 'device_unavailable'],
                 tool: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f'[{tool}:{code}] {message}')
        self.code, self.tool, self.message, self.details = code, tool, message, details or {}

    def to_dict(self) -> dict[str, Any]:
        return {'code': self.code, 'tool': self.tool, 'message': self.message, 'details': self.details}


class RunContext(BaseModel):
    """Per-run settings the engine passes to execute(); never shared state between tools."""
    model_config = ConfigDict(frozen=True)
    seed: int = 42
    device: DeviceName | Literal['auto'] = 'auto'
    node_id: str = ''


class ToolResult(BaseModel):
    outputs: dict[str, Any]
    inference_ms: float
    device: str
    weights: Literal['checkpoint', 'fitted_on_input', 'mock_init']
    diagnostics: dict[str, Any] = Field(default_factory=dict)


InT = TypeVar('InT', bound=BaseModel)
ParT = TypeVar('ParT', bound=BaseModel)


class BaseDLTool(abc.ABC, Generic[InT, ParT]):
    """Lifecycle and safety shared by all tools. Subclasses only define schemas, the model and _infer()."""

    spec: ClassVar[ToolSpec]
    Input: ClassVar[type[BaseModel]]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        try:
            self.params: ParT = self.Params.model_validate(params or {})  # type: ignore[assignment]
        except ValidationError as exc:
            raise ToolError('invalid_params', self.spec.name, 'Parámetros inválidos', {'errors': exc.errors(include_url=False)}) from exc
        self.model: Any = None
        self.device: str = 'cpu'
        self.weights: Literal['checkpoint', 'fitted_on_input', 'mock_init'] = 'mock_init'
        self.warm = False

    # lifecycle ------------------------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, context: RunContext) -> None:
        """Lazy load: build the nn.Module, restore weights if a checkpoint exists, else a seeded safe init."""
        if self.loaded:
            return
        self.device = resolve_device(context.device, self.spec.capabilities.devices)
        self.model, self.weights = self._build(context)
        log.info('%s loaded on %s (%s)', self.spec.name, self.device, self.weights)

    def warmup(self, context: RunContext) -> None:
        if not self.loaded:
            self.load(context)
        if not self.warm:
            self._warmup()
            self.warm = True

    def unload(self) -> None:
        """Explicit release: drop references, flush the device cache and collect garbage."""
        if self.model is not None:
            release(self.model, self.device)
        self.model, self.warm = None, False

    def __enter__(self) -> 'BaseDLTool[InT, ParT]':
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    # execution ------------------------------------------------------------------------------------
    def execute(self, payload: dict[str, Any], context: RunContext | None = None) -> ToolResult:
        ctx = context or RunContext()
        try:
            data = self.Input.model_validate(payload)
        except ValidationError as exc:
            raise ToolError('invalid_input', self.spec.name, 'Entrada inválida', {'errors': exc.errors(include_url=False)}) from exc
        self.warmup(ctx)
        start = time.perf_counter()
        try:
            outputs, diagnostics = self._infer(data, ctx)  # type: ignore[arg-type]
        except ToolError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise ToolError('execution_failed', self.spec.name, str(exc)) from exc
        elapsed = (time.perf_counter() - start) * 1000
        missing = [p.name for p in self.spec.outputs if p.required and p.name not in outputs]
        if missing:
            raise ToolError('execution_failed', self.spec.name, f'Salidas no producidas: {missing}')
        return ToolResult(outputs=outputs, inference_ms=round(elapsed, 3), device=self.device, weights=self.weights, diagnostics=diagnostics)

    # subclass hooks -------------------------------------------------------------------------------
    @abc.abstractmethod
    def _build(self, context: RunContext) -> tuple[Any, Literal['checkpoint', 'fitted_on_input', 'mock_init']]: ...

    def _warmup(self) -> None:
        """Optional dummy forward pass (allocates kernels once)."""

    @abc.abstractmethod
    def _infer(self, data: InT, context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]: ...
