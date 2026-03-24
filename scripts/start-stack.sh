#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/.runtime"
LOG_DIR="$REPO_ROOT/output/logs"
VENV_DIR="${HOME}/.venvs/auto-research-scrapling"
SCRAPLING_PORT="${SCRAPLING_PORT:-8010}"
SCRAPLING_BASE="http://127.0.0.1:${SCRAPLING_PORT}"
VAULT_ROOT="${VAULT_ROOT:-/mnt/c/Users/User/Documents/AutoResearchVault}"
LMSTUDIO_LOCAL_PORT="${LMSTUDIO_LOCAL_PORT:-1234}"
PI_DIR="${HOME}/.pi/agent"

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib-host-endpoints.sh"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$REPO_ROOT/output/sources" "$REPO_ROOT/output/research" "$REPO_ROOT/output/notes" "$REPO_ROOT/knowledge/logs" "$REPO_ROOT/staging/tooling" "$REPO_ROOT/staging/skills-candidates" "$REPO_ROOT/sandbox/rd-agent/in" "$REPO_ROOT/sandbox/rd-agent/out" "$VAULT_ROOT/10_Research/AutoResearch" "$VAULT_ROOT/90_Logs"

WINDOWS_BRIDGE_TARGET_HOST="${WINDOWS_BRIDGE_TARGET_HOST:-$(default_windows_host)}"

ensure_windows_loopback() {
  local name="$1"
  local listen_port="$2"
  local target_port="$3"
  local path="$4"
  local pidfile="$RUNTIME_DIR/${name}-bridge.pid"
  local logfile="$LOG_DIR/${name}-bridge.log"

  if is_http_ok "http://127.0.0.1:${listen_port}${path}"; then
    return 0
  fi

  if ! command -v curl.exe >/dev/null 2>&1; then
    return 1
  fi

  if ! curl.exe --max-time 5 -fsS "http://${WINDOWS_BRIDGE_TARGET_HOST}:${target_port}${path}" >/dev/null 2>&1; then
    return 1
  fi

  local old_pid=""
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
      sleep 1
      if is_http_ok "http://127.0.0.1:${listen_port}${path}"; then
        return 0
      fi
      kill "$old_pid" >/dev/null 2>&1 || true
    fi
    rm -f "$pidfile"
  fi

  nohup python3 "$SCRIPT_DIR/windows-http-bridge.py" \
    --name "$name" \
    --listen-host 127.0.0.1 \
    --listen-port "$listen_port" \
    --target-host "$WINDOWS_BRIDGE_TARGET_HOST" \
    --target-port "$target_port" \
    >"$logfile" 2>&1 &
  echo $! > "$pidfile"
  sleep 2

  is_http_ok "http://127.0.0.1:${listen_port}${path}"
}

if ensure_windows_loopback "ollama" "11434" "11434" "/api/tags"; then
  OLLAMA_BASE="http://127.0.0.1:11434"
else
  OLLAMA_BASE="$(resolve_ollama_v1_base)"
fi

if is_http_ok "http://127.0.0.1:1234/v1/models"; then
  LMSTUDIO_BASE="http://127.0.0.1:1234"
elif ensure_windows_loopback "lmstudio" "$LMSTUDIO_LOCAL_PORT" "1234" "/v1/models"; then
  LMSTUDIO_BASE="http://127.0.0.1:${LMSTUDIO_LOCAL_PORT}"
else
  LMSTUDIO_BASE="$(resolve_lmstudio_v1_base)"
fi

if ! is_http_ok "${OLLAMA_BASE}/api/tags"; then
  echo "[error] Ollama is not reachable at ${OLLAMA_BASE}" >&2
  exit 1
fi

echo "[info] Ollama endpoint: ${OLLAMA_BASE}"

if is_http_ok "${LMSTUDIO_BASE}/v1/models"; then
  echo "[info] LM Studio server is reachable at ${LMSTUDIO_BASE}"
else
  echo "[warn] LM Studio server is not reachable at ${LMSTUDIO_BASE}"
fi

scrapling_pid=""
if [[ -f "$RUNTIME_DIR/scrapling.pid" ]]; then
  scrapling_pid="$(cat "$RUNTIME_DIR/scrapling.pid" 2>/dev/null || true)"
fi
if [[ -n "$scrapling_pid" ]] && kill -0 "$scrapling_pid" >/dev/null 2>&1; then
  echo "[info] Scrapling already running with pid $scrapling_pid"
else
  rm -f "$RUNTIME_DIR/scrapling.pid"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] Scrapling venv not found at $VENV_DIR. Run bootstrap-wsl.sh first." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  nohup scrapling mcp --http --host 127.0.0.1 --port "$SCRAPLING_PORT" >"$LOG_DIR/scrapling.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/scrapling.pid"
  deactivate
  sleep 5
fi

if command -v mcporter >/dev/null 2>&1; then
  MCPORTER=(mcporter)
else
  MCPORTER=(npx mcporter)
fi

MCP_URL=""
TMP_MCPORTER_CONFIG="$RUNTIME_DIR/mcporter.validate.json"

write_mcporter_config() {
  local config_path="$1"
  local base_url="$2"
  cat > "$config_path" <<EOF
{
  "mcpServers": {
    "scrapling": {
      "baseUrl": "${base_url}"
    }
  }
}
EOF
}

for candidate in "${SCRAPLING_BASE}/mcp" "${SCRAPLING_BASE}"; do
  write_mcporter_config "$TMP_MCPORTER_CONFIG" "$candidate"
  if "${MCPORTER[@]}" --config "$TMP_MCPORTER_CONFIG" list scrapling --schema >/dev/null 2>&1; then
    MCP_URL="$candidate"
    break
  fi
done

rm -f "$TMP_MCPORTER_CONFIG"

if [[ -z "$MCP_URL" ]]; then
  echo "[error] Unable to discover Scrapling MCP endpoint on ${SCRAPLING_BASE}" >&2
  exit 1
fi

cat > "$REPO_ROOT/config/mcporter.json" <<EOF
{
  "mcpServers": {
    "scrapling": {
      "baseUrl": "${MCP_URL}"
    }
  }
}
EOF

cat > "$RUNTIME_DIR/service-endpoints.env" <<EOF
export OLLAMA_BASE="${OLLAMA_BASE}"
export LMSTUDIO_BASE="${LMSTUDIO_BASE}"
EOF

if [[ -d "$PI_DIR" ]]; then
  cat > "$PI_DIR/models.json" <<EOF
{
  "providers": {
    "ollama": {
      "baseUrl": "${OLLAMA_BASE}/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.5:9b",
          "name": "qwen3.5:9b"
        }
      ]
    },
    "lmstudio": {
      "baseUrl": "${LMSTUDIO_BASE}/v1",
      "api": "openai-completions",
      "apiKey": "lmstudio",
      "authHeader": false,
      "models": [
        {
          "id": "nvidia/nemotron-3-nano",
          "name": "nvidia/nemotron-3-nano"
        },
        {
          "id": "text-embedding-nomic-embed-text-v1.5",
          "name": "text-embedding-nomic-embed-text-v1.5"
        }
      ]
    }
  }
}
EOF
fi

echo "[ok] stack ready"
echo "Scrapling MCP URL: ${MCP_URL}"
