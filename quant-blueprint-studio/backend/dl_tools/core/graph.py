"""Minimal Blueprint executor for DL tools: validate ports, topological order, run, always unload.

Graph JSON:
  {"inputs": {"<name>": value},                       # graph-level values, referenced as source "$<name>"
   "nodes": [{"id": "f", "tool": "dl_tabular_forecast", "params": {...}}],
   "edges": [{"source": "$series", "target": "f", "target_port": "series"},
             {"source": "f", "source_port": "residuals", "target": "a", "target_port": "data"}]}
Tools never see each other: values travel only through ports.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from . import registry
from .contracts import BaseDLTool, RunContext, ToolError
from .memory import device_bytes, flush


class Edge(BaseModel):
    source: str
    source_port: str | None = None
    target: str
    target_port: str


class Node(BaseModel):
    id: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class Graph(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    nodes: list[Node]
    edges: list[Edge]


def validate(graph: Graph) -> list[str]:
    """Port-level checks before anything is loaded. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    nodes = {n.id: n for n in graph.nodes}
    if len(nodes) != len(graph.nodes):
        problems.append('Ids de nodo duplicados.')
    seen: set[tuple[str, str]] = set()
    for e in graph.edges:
        if e.target not in nodes:
            problems.append(f'Destino inexistente: {e.target}')
            continue
        spec = registry.get(nodes[e.target].tool).spec
        if e.target_port not in {p.name for p in spec.inputs}:
            problems.append(f'{e.target}: puerto de entrada inexistente {e.target_port}')
            continue
        if (e.target, e.target_port) in seen:
            problems.append(f'{e.target}.{e.target_port}: sólo una conexión por puerto')
        seen.add((e.target, e.target_port))
        if e.source.startswith('$'):
            if e.source[1:] not in graph.inputs:
                problems.append(f'Entrada de grafo inexistente: {e.source}')
            continue
        if e.source not in nodes:
            problems.append(f'Origen inexistente: {e.source}')
            continue
        out = {p.name: p for p in registry.get(nodes[e.source].tool).spec.outputs}
        if e.source_port not in out:
            problems.append(f'{e.source}: puerto de salida inexistente {e.source_port}')
            continue
        want = spec.input(e.target_port).type
        if want != 'any' and out[e.source_port].type != want:
            problems.append(f'Tipos incompatibles {e.source}.{e.source_port} ({out[e.source_port].type}) → {e.target}.{e.target_port} ({want})')
    for n in graph.nodes:
        wired = {e.target_port for e in graph.edges if e.target == n.id}
        for p in registry.get(n.tool).spec.inputs:
            if p.required and p.name not in wired:
                problems.append(f'{n.id}: falta la entrada obligatoria {p.name}')
    return problems


def order(graph: Graph) -> list[str]:
    deps = {n.id: {e.source for e in graph.edges if e.target == n.id and not e.source.startswith('$')} for n in graph.nodes}
    ready = deque(k for k, v in deps.items() if not v)
    out: list[str] = []
    while ready:
        k = ready.popleft()
        out.append(k)
        for m, v in deps.items():
            if k in v:
                v.discard(k)
                if not v:
                    ready.append(m)
    if len(out) != len(deps):
        raise ValueError('El grafo tiene un ciclo.')
    return out


def run(graph: Graph | dict[str, Any], context: RunContext | None = None) -> dict[str, Any]:
    """Execute the graph. Every instantiated tool is unloaded in `finally`, also on failure."""
    g = graph if isinstance(graph, Graph) else Graph.model_validate(graph)
    problems = validate(g)
    if problems:
        raise ValueError('Grafo inválido: ' + '; '.join(problems))
    ctx = context or RunContext()
    nodes = {n.id: n for n in g.nodes}
    live: dict[str, BaseDLTool[Any, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {'nodes': {}, 'memory': {}}
    try:
        for key in order(g):
            node = nodes[key]
            payload = {e.target_port: (g.inputs[e.source[1:]] if e.source.startswith('$') else results[e.source][e.source_port or ''])
                       for e in g.edges if e.target == key}
            tool = registry.get(node.tool)(node.params)
            live[key] = tool
            result = tool.execute(payload, RunContext(seed=ctx.seed, device=ctx.device, node_id=key))
            results[key] = result.outputs
            report['nodes'][key] = {'tool': node.tool, 'inference_ms': result.inference_ms, 'device': result.device,
                                    'weights': result.weights, 'diagnostics': result.diagnostics}
            report['memory'].setdefault('peak_bytes', 0)
            report['memory']['peak_bytes'] = max(report['memory']['peak_bytes'], device_bytes(result.device))
    except ToolError as exc:
        report['error'] = exc.to_dict()
        raise
    finally:
        devices = {t.device for t in live.values()}
        for t in live.values():
            t.unload()
        for d in devices:
            flush(d)
        report['memory']['unloaded'] = sorted(live)
        report['memory']['after_bytes'] = {d: device_bytes(d) for d in devices}
        report['memory']['all_released'] = all(not t.loaded for t in live.values())
    report['outputs'] = results
    return report
