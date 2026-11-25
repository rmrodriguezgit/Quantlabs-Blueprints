"""Local FastAPI bridge. CPU work runs off the event loop, one graph at a time."""
import asyncio
import os
import threading
try:
    import xgboost  # noqa: F401  antes que torch: evita el choque de libomp en macOS (ver backend/__init__.py)
except ImportError:
    pass
import torch

# Small tabular networks benefit from bounded CPU parallelism on desktop hosts.
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
try:
    from .engine.dag_runner import DAGRunner
    from .engine.nodes import polymarket_nodes as polymarket
    from .brain import fixer as brain_fixer
    from .brain import explain as brain_explain, llm as brain_llm, planner as brain_planner, store as brain_store, tutor as brain_tutor
except ImportError:
    from engine.dag_runner import DAGRunner
    from engine.nodes import polymarket_nodes as polymarket
    from brain import fixer as brain_fixer
    from brain import explain as brain_explain, llm as brain_llm, planner as brain_planner, store as brain_store, tutor as brain_tutor


import json as _json
from pathlib import Path as _Path
NODE_KINDS = tuple(_json.loads((_Path(__file__).resolve().parents[1]/'src/renderer/data/node-contracts.json').read_text())['nodes'])
NodeKind = Literal[NODE_KINDS]  # type: ignore[valid-type]


class NodeData(BaseModel):
    model_config = ConfigDict(extra='ignore')
    kind: NodeKind  # derivado de node-contracts.json: una herramienta nueva no requiere editar esta lista
    label: str = ''
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Node(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    data: NodeData


class Edge(BaseModel):
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class Graph(BaseModel):
    nodes: list[Node] = Field(min_length=1, max_length=50)
    edges: list[Edge] = Field(default_factory=list, max_length=150)


app = FastAPI(title='Quant Blueprint Engine', version='0.8.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
execution_lock = threading.Lock()


class PolymarketQuery(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    up_outcome: str = 'auto'


@app.post('/api/polymarket/search')
def polymarket_search(request: PolymarketQuery) -> dict[str, Any]:
    try:
        return {'results': polymarket.search_markets(request.query, 10)}
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post('/api/polymarket/resolve')
def polymarket_resolve(request: PolymarketQuery) -> dict[str, Any]:
    try:
        return {'market': polymarket.market_meta(request.query, request.up_outcome)}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class LearningKey(BaseModel):
    key: str = Field(min_length=1, max_length=120)


@app.post('/api/polymarket/learning/reset')
def polymarket_learning_reset(request: LearningKey) -> dict[str, Any]:
    return {'reset': polymarket.reset_learning(request.key)}


@app.get('/api/health')
def health() -> dict[str, Any]:
    return {'status': 'ready', 'pid': os.getpid()}


def run(graph: Graph) -> dict[str, Any]:
    if not execution_lock.acquire(blocking=False):
        raise HTTPException(409, 'Engine busy. Wait for the current run.')
    try:
        return DAGRunner(graph.model_dump()).execute()
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        execution_lock.release()


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    engine: Literal['auto', 'ollama', 'rules'] = 'auto'
    catalog: dict[str, dict[str, Any]]


class ExplainRequest(BaseModel):
    report: dict[str, Any]
    engine: Literal['auto', 'ollama', 'rules'] = 'auto'


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    engine: Literal['auto', 'ollama', 'rules'] = 'auto'
    node_type: str | None = None
    catalog: dict[str, dict[str, Any]]


class FeedbackRequest(BaseModel):
    plan_id: str = Field(min_length=1, max_length=40)
    rating: Literal[-1, 0, 1]


class RequestStatus(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    status: Literal['pending', 'planned', 'discarded']


@app.post('/api/brain/ask')
def brain_ask(request: AskRequest) -> dict[str, Any]:
    try:
        return brain_tutor.ask(request.question, request.catalog, request.engine, request.node_type)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/brain/ping')
def brain_ping() -> dict[str, Any]:
    return brain_llm.ping()


@app.get('/api/brain/memory')
def brain_memory() -> dict[str, Any]:
    return brain_store.memory()


@app.post('/api/brain/feedback')
def brain_feedback(request: FeedbackRequest) -> dict[str, Any]:
    try:
        return brain_store.rate(request.plan_id, request.rating)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/brain/request-status')
def brain_request_status(request: RequestStatus) -> dict[str, Any]:
    try:
        return brain_store.set_request_status(request.id, request.status)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/validate')
def validate(graph: Graph) -> dict[str, Any]:
    """Engine-side structural check (pins, cycles, required inputs) without executing."""
    return brain_planner.structural_check(graph.model_dump())


@app.get('/api/brain/status')
def brain_status() -> dict[str, Any]:
    return brain_llm.status()


@app.post('/api/brain/plan')
def brain_plan(request: PlanRequest) -> dict[str, Any]:
    try:
        return brain_planner.plan(request.prompt, request.catalog, request.engine)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class FixRequest(BaseModel):
    graph: dict[str, Any]
    error: str | None = Field(default=None, max_length=4000)
    issues: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    catalog: dict[str, dict[str, Any]]
    engine: Literal['auto', 'ollama', 'rules'] = 'auto'


@app.post('/api/brain/fix')
def brain_fix(request: FixRequest) -> dict[str, Any]:
    try:
        return brain_fixer.fix(request.graph, request.error, request.issues, request.catalog, request.engine)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/brain/explain')
def brain_explain_result(request: ExplainRequest) -> dict[str, Any]:
    try:
        return brain_explain.explain(request.report, request.engine)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/execute')
def execute(graph: Graph) -> dict[str, Any]:
    return run(graph)


@app.websocket('/ws/execute')
async def execute_ws(socket: WebSocket) -> None:
    origin = socket.headers.get('origin')
    if origin not in (None, 'http://127.0.0.1:5173', 'http://localhost:5173'):
        await socket.close(code=1008)
        return
    await socket.accept()
    acquired = False
    try:
        graph = Graph.model_validate(await socket.receive_json())
        acquired = execution_lock.acquire(blocking=False)
        if not acquired:
            await socket.send_json({'event': 'error', 'detail': 'Engine busy.'})
            return
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        def progress(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)
        async def worker() -> None:
            try:
                result = await asyncio.to_thread(DAGRunner(graph.model_dump()).execute, progress)
                await events.put({'event': 'result', 'result': result})
            except Exception as exc:
                await events.put({'event': 'error', 'detail': str(exc)})
        task = asyncio.create_task(worker())
        try:
            while True:
                event = await events.get()
                await socket.send_json(event)
                if event.get('event') in ('result', 'error'):
                    break
        finally:
            # Disconnecting a client does not stop its thread. Keep the lock until it finishes.
            await asyncio.shield(task)
    except WebSocketDisconnect:
        pass
    except ValueError as exc:
        await socket.send_json({'event': 'error', 'detail': str(exc)})
    finally:
        if acquired:
            execution_lock.release()
        try:
            await socket.close()
        except RuntimeError:
            pass


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('QL_ENGINE_PORT', '8000')))
