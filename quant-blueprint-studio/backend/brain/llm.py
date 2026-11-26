"""Minimal local Ollama client. Loopback only; replies are untrusted JSON that callers validate."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

URL = os.environ.get('QL_BRAIN_OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
PREFERRED = ['qwen3:8b', 'qwen3.5:4b', 'qwen3.5-4b-ctx16k:latest', 'qwen2.5:7b-instruct', 'qwen2.5:3b']
_STATUS: dict[str, Any] = {'at': 0.0, 'value': None}


class LLMError(RuntimeError):
    pass


def _local(url: str) -> bool:
    return urllib.parse.urlparse(url).hostname in ('127.0.0.1', 'localhost', '::1') or os.environ.get('QL_BRAIN_ALLOW_REMOTE') == '1'


def _http(path: str, body: dict[str, Any] | None = None, timeout: float = 3) -> dict[str, Any]:
    if not _local(URL):
        raise LLMError('Ollama debe estar en 127.0.0.1 (QL_BRAIN_ALLOW_REMOTE=1 para otro host).')
    request = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'}, method='GET' if body is None else 'POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LLMError(f'Ollama no responde en {URL}: {exc}') from exc


def status(force: bool = False) -> dict[str, Any]:
    """Available model, cached 30 s so the UI can poll cheaply."""
    if not force and _STATUS['value'] is not None and time.time() - _STATUS['at'] < 30:
        return _STATUS['value']
    try:
        names = [m['name'] for m in _http('/api/tags').get('models', [])]
        wanted = os.environ.get('QL_BRAIN_MODEL')
        model = wanted if wanted in names else next((m for m in PREFERRED if m in names), next((n for n in names if 'qwen' in n), names[0] if names else None))
        value = {'available': bool(model), 'model': model, 'models': names, 'url': URL, 'reason': None if model else 'Ollama sin modelos: ollama pull qwen3.5:4b'}
    except LLMError as exc:
        value = {'available': False, 'model': None, 'models': [], 'url': URL, 'reason': str(exc)}
    _STATUS.update(at=time.time(), value=value)
    return value


def chat_json(system: str, user: str, schema: dict[str, Any] | None = None, timeout: float = 150) -> tuple[dict[str, Any], str]:
    info = status()
    if not info['available']:
        raise LLMError(info['reason'] or 'Sin modelo local.')
    body = {'model': info['model'], 'stream': False, 'think': False, 'format': schema or 'json',
            'options': {'temperature': 0.2, 'num_ctx': 8192},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
    try:
        reply = _http('/api/chat', body, timeout)
    except LLMError:
        body['format'] = 'json'      # some runners reject JSON schemas; retry with plain JSON mode
        body.pop('think', None)
        reply = _http('/api/chat', body, timeout)
    text = reply.get('message', {}).get('content', '')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    match = re.search(r'\{.*\}', text, flags=re.S)
    try:
        value = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError as exc:
        raise LLMError('El modelo no devolvió JSON válido.') from exc
    if not isinstance(value, dict):
        raise LLMError('El modelo no devolvió un objeto JSON.')
    return value, info['model']


def ping() -> dict[str, Any]:
    """Real check: server answers, a model exists and it generates a few tokens (semaphore in the UI)."""
    info = status(force=True)
    base = {'url': URL, 'model': info['model'], 'seconds': None, 'reply': None, 'error': info['reason']}
    if not info['models'] and info['reason'] and 'no responde' in info['reason']:
        return {**base, 'state': 'offline'}
    if not info['available']:
        return {**base, 'state': 'no_model'}
    started = time.monotonic()
    try:
        reply = _http('/api/generate', {'model': info['model'], 'prompt': 'Responde solo: ok', 'stream': False, 'think': False,
                                        'options': {'num_predict': 4, 'temperature': 0}}, timeout=60)
    except LLMError as exc:
        return {**base, 'state': 'offline', 'error': str(exc)}
    seconds = round(time.monotonic() - started, 2)
    text = re.sub(r'<think>.*?</think>', '', str(reply.get('response', '')), flags=re.S).strip()[:40]
    return {**base, 'state': 'ok' if seconds < 15 else 'slow', 'seconds': seconds, 'reply': text, 'error': None}
