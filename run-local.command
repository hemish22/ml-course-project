#!/bin/bash
# Run from Terminal or double-click in Finder. Installs missing local dependencies.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$project_dir"
mkdir -p .runtime
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="$project_dir/.runtime/matplotlib"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
api_port="${NEEDLE_API_PORT:-8000}"
web_port="${NEEDLE_WEB_PORT:-3000}"
export NEXT_PUBLIC_API_URL="http://127.0.0.1:$api_port"

if [[ ! -x .venv/bin/python ]]; then
  python_bin="${PYTHON:-}"
  if [[ -z "$python_bin" ]]; then
    for candidate in python3.11 python3.10 python3.12 python3; do
      if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        python_bin="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$python_bin" ]]; then
    echo 'Python 3.10+ is required. Set PYTHON to its executable and run again.'
    exit 1
  fi
  "$python_bin" -m venv .venv
fi

if ! .venv/bin/python -c 'import importlib.util,sys; names=["torch","open_clip","faster_whisper","faiss","sentence_transformers","numpy","PIL","pandas","sklearn","scipy","matplotlib","tqdm","yaml","fastapi","uvicorn","httpx"]; missing=[n for n in names if importlib.util.find_spec(n) is None]; print("Missing Python packages:", ", ".join(missing)) if missing else None; sys.exit(bool(missing))'; then
  echo 'Installing Python dependencies in .venv (internet required).'
  .venv/bin/python -m pip install --cache-dir .runtime/pip -r requirements.txt
fi

if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo 'Node.js and npm are required to start the web interface.'
  exit 1
fi

lock_hash="$(.venv/bin/python -c 'import hashlib,pathlib; print(hashlib.sha256(pathlib.Path("web/package-lock.json").read_bytes()).hexdigest())')"
if [[ ! -x web/node_modules/.bin/next ]] || [[ ! -f .runtime/web-lock ]] || [[ "$(cat .runtime/web-lock)" != "$lock_hash" ]]; then
  echo 'Installing the locked frontend dependencies (internet required).'
  (cd web && npm ci --cache ../.runtime/npm --no-audit --no-fund)
  printf '%s\n' "$lock_hash" > .runtime/web-lock
fi

api_pid=''
web_pid=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$api_pid" ]] || kill "$api_pid" 2>/dev/null || true
  [[ -z "$web_pid" ]] || kill "$web_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

.venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port "$api_port" > .runtime/api.log 2>&1 &
api_pid=$!
(cd web && exec node node_modules/next/dist/bin/next dev --hostname 127.0.0.1 --port "$web_port") > .runtime/web.log 2>&1 &
web_pid=$!

echo "Starting Needle. First startup loads the query encoders."
echo "Web: http://127.0.0.1:$web_port | API: http://127.0.0.1:$api_port/docs"
echo 'Logs: .runtime/api.log and .runtime/web.log. Press Ctrl+C to stop both servers.'

if ! .venv/bin/python - "$api_port" "$web_port" "$api_pid" "$web_pid" <<'PY'
import os
import sys
import time
from urllib.request import urlopen

api_port, web_port, api_pid, web_pid = map(int, sys.argv[1:])
pending = {f'http://127.0.0.1:{api_port}/api/config', f'http://127.0.0.1:{web_port}'}
deadline = time.monotonic() + float(os.environ.get('NEEDLE_STARTUP_TIMEOUT', '300'))
while pending and time.monotonic() < deadline:
    for pid in [api_pid, web_pid]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            sys.exit('A server exited during startup; see the logs below.')
    for url in list(pending):
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    pending.remove(url)
        except OSError:
            pass
    if pending:
        time.sleep(.5)
if pending:
    sys.exit('Startup timed out. Initial model downloads may need more time; see the logs below.')
print('Needle is ready.')
PY
then
  tail -n 20 .runtime/api.log .runtime/web.log
  exit 1
fi

if command -v open >/dev/null; then
  open "http://127.0.0.1:$web_port" || true
fi
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
  sleep 2
done
echo 'A server stopped. See .runtime/api.log and .runtime/web.log.'
exit 1
