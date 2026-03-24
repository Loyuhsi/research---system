#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/.runtime"

stop_by_pidfile() {
  local name="$1"
  local pidfile="$RUNTIME_DIR/${name}.pid"
  if [[ ! -f "$pidfile" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "[info] Stopping $name (pid $pid)"
    kill "$pid" 2>/dev/null || true
    # Wait briefly for graceful shutdown
    local i=0
    while [[ $i -lt 10 ]] && kill -0 "$pid" >/dev/null 2>&1; do
      sleep 0.5
      ((i++))
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "[warn] $name (pid $pid) did not exit gracefully, sending SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pidfile"
}

stop_by_pidfile "scrapling"
stop_by_pidfile "ollama-bridge"
stop_by_pidfile "lmstudio-bridge"

echo "[ok] stack stopped"
