#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v node >/dev/null || { echo 'Install Node.js 22+.'; exit 1; }
node -e 'if (+process.versions.node.split(".")[0] < 22) process.exit(1)' || { echo 'Node.js 22+ required.'; exit 1; }
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))'; then PYTHON="$candidate"; break; fi
done
[ -n "$PYTHON" ] || { echo 'Install Python 3.10–3.13 (PyTorch wheel compatibility).'; exit 1; }
[ -d .venv ] || "$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
[ -d node_modules ] || npm install
if [ -z "${QL_ENGINE_PORT:-}" ]; then
  QL_ENGINE_PORT="$(python - <<'PY'
import socket
for port in range(8000, 8100):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', port)); print(port); break
        except OSError:
            continue
PY
)"
fi
export QL_ENGINE_PORT
export VITE_ENGINE_PORT="$QL_ENGINE_PORT"
echo "Python engine on 127.0.0.1:${QL_ENGINE_PORT}"
python backend/server.py &
ENGINE_PID=$!
trap 'kill "$ENGINE_PID" 2>/dev/null || true' EXIT INT TERM
python scripts/wait_backend.py "$ENGINE_PID"
npm run electron:dev
