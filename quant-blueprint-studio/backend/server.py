"""Local FastAPI bridge. CPU work runs off the event loop, one graph at a time."""
import asyncio
import os
import threading
import torch

# Small tabular networks benefit from bounded CPU parallelism on desktop hosts.
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
try:
    from .engine.dag_runner import DAGRunner
except ImportError:
    from engine.dag_runner import DAGRunner


class NodeData(BaseModel):
    model_config = ConfigDict(extra='ignore')
    kind: Literal['source', 'features', 'prepare', 'model', 'backtest', 'config', 'network_input', 'layer', 'network_output', 'merge', 'plot', 'portfolio_data', 'portfolio', 'risk', 'study']
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


app = FastAPI(title='Quant Blueprint Engine', version='0.5.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
execution_lock = threading.Lock()


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
    uvicorn.run(app, host='127.0.0.1', port=8000)
