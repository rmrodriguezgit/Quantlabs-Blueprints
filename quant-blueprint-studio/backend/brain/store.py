"""Local brain memory: plan history, ratings and missing-tool requests (atomic JSON writes).

Stored in ``QL_BRAIN_DIR`` or ``<repo>/.quantlabs/brain`` (git-ignored). Never leaves the machine.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
LIMITS = {'history': 200, 'feedback': 300, 'requests': 200}
STATUSES = ('pending', 'planned', 'discarded')


def root() -> Path:
    path = Path(os.environ.get('QL_BRAIN_DIR') or Path(__file__).resolve().parents[3] / '.quantlabs' / 'brain')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(name: str, default: Any) -> Any:
    try:
        return json.loads((root() / name).read_text())
    except (OSError, ValueError):
        return default


def _write(name: str, value: Any) -> None:
    target = root() / name
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=name, suffix='.tmp')
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, target)


def remember(plan: dict[str, Any], prompt: str) -> str:
    """Record a built plan; returns its id for later rating."""
    plan_id = uuid.uuid4().hex[:12]
    entry = {'id': plan_id, 'at': time.time(), 'prompt': prompt[:2000], 'title': plan.get('title'), 'engine': plan.get('engine'),
             'model': plan.get('model'), 'steps': [s['tool'] for s in plan.get('steps', [])], 'graph': plan.get('graph'), 'ok': plan.get('check', {}).get('ok')}
    with _LOCK:
        history = _read('history.json', [])[-(LIMITS['history'] - 1):]
        _write('history.json', [*history, entry])
        requests = _read('tool-requests.json', [])
        for missing in plan.get('missing', []):
            name = str(missing.get('name', '')).strip()[:120]
            if not name:
                continue
            found = next((r for r in requests if r['name'].lower() == name.lower()), None)
            if found:
                found['count'] += 1
                found['prompts'] = [*found['prompts'][-4:], prompt[:300]]
            else:
                requests.append({'id': uuid.uuid4().hex[:10], 'name': name, 'purpose': str(missing.get('purpose', ''))[:500],
                                 'count': 1, 'status': 'pending', 'prompts': [prompt[:300]], 'at': time.time()})
        _write('tool-requests.json', requests[-LIMITS['requests']:])
    return plan_id


def rate(plan_id: str, rating: int) -> dict[str, Any]:
    if rating not in (-1, 0, 1):
        raise ValueError('La valoración es -1, 0 o 1.')
    with _LOCK:
        history = _read('history.json', [])
        entry = next((h for h in history if h['id'] == plan_id), None)
        if not entry:
            raise ValueError('No existe ese blueprint en la memoria.')
        entry['rating'] = rating
        _write('history.json', history)
    return {'id': plan_id, 'rating': rating}


def set_request_status(request_id: str, status: str) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError('Estado: pending, planned o discarded.')
    with _LOCK:
        requests = _read('tool-requests.json', [])
        item = next((r for r in requests if r['id'] == request_id), None)
        if not item:
            raise ValueError('No existe esa ficha.')
        item['status'] = status
        _write('tool-requests.json', requests)
    return item


def liked() -> list[dict[str, Any]]:
    """Blueprints rated 👍 with a valid graph: they feed future retrieval."""
    return [h for h in _read('history.json', []) if h.get('rating') == 1 and h.get('ok') and h.get('graph')]


def memory() -> dict[str, Any]:
    history = _read('history.json', [])
    return {'history': [{k: v for k, v in h.items() if k != 'graph'} for h in reversed(history[-40:])],
            'requests': sorted(_read('tool-requests.json', []), key=lambda r: (r['status'] != 'pending', -r['count'])),
            'liked': len(liked()), 'path': str(root())}
