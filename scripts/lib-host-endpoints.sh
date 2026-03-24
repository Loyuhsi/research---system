#!/usr/bin/env bash
set -euo pipefail

# --- Load .env if present ---
# REPO_ROOT must be set by the calling script before sourcing this file.
if [[ -n "${REPO_ROOT:-}" && -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# --- Shared helpers ---

die() {
  echo "[error] $*" >&2
  exit 1
}

warn() {
  echo "[warn] $*" >&2
}

log_debug() {
  if [[ "${AUTO_RESEARCH_DEBUG:-0}" == "1" ]]; then
    echo "[debug] $*" >&2
  fi
}

retry_cmd() {
  local max_attempts="${FETCH_MAX_RETRIES:-3}"
  local delay="${FETCH_RETRY_DELAY:-5}"
  local attempt=1
  while [[ $attempt -le $max_attempts ]]; do
    if "$@"; then
      return 0
    fi
    if [[ $attempt -lt $max_attempts ]]; then
      warn "Attempt $attempt/$max_attempts failed, waiting ${delay}s..."
      sleep "$delay"
      delay=$((delay * 2))
    fi
    ((attempt++))
  done
  return 1
}

unique_lines() {
  awk 'NF && !seen[$0]++'
}

windows_host_candidates() {
  {
    if [[ -n "${WINDOWS_HOST_IP:-}" ]]; then
      printf '%s\n' "$WINDOWS_HOST_IP"
    fi

    printf '%s\n' "localhost" "127.0.0.1"

    local gateway=""
    gateway="$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')"
    if [[ -n "$gateway" ]]; then
      printf '%s\n' "$gateway"
    fi

    local nameserver=""
    nameserver="$(awk '/^nameserver[[:space:]]+/ {print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"
    if [[ -n "$nameserver" && "$nameserver" != "127.0.0.1" ]]; then
      printf '%s\n' "$nameserver"
    fi
  } | unique_lines
}

probe_http_base() {
  local port="$1"
  local path="$2"
  local host

  while IFS= read -r host; do
    log_debug "Probing http://${host}:${port}${path}"
    if curl --max-time 3 -fsS "http://${host}:${port}${path}" >/dev/null 2>&1; then
      log_debug "Found service at http://${host}:${port}"
      printf 'http://%s:%s\n' "$host" "$port"
      return 0
    fi
  done < <(windows_host_candidates)

  log_debug "No service found on port $port"
  return 1
}

default_windows_host() {
  local gateway=""
  gateway="$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')"
  if [[ -n "$gateway" ]]; then
    printf '%s\n' "$gateway"
  else
    printf '%s\n' "localhost"
  fi
}

resolve_windows_service_base() {
  local port="$1"
  local path="$2"
  local explicit_base="${3:-}"

  if [[ -n "$explicit_base" ]]; then
    printf '%s\n' "${explicit_base%/}"
    return 0
  fi

  if probe_http_base "$port" "$path"; then
    return 0
  fi

  printf 'http://%s:%s\n' "$(default_windows_host)" "$port"
}

resolve_ollama_v1_base() {
  resolve_windows_service_base "11434" "/api/tags" "${WINDOWS_OLLAMA_BASE_URL:-}"
  return 0
}

resolve_lmstudio_v1_base() {
  local ollama_base=""
  local explicit_base="${WINDOWS_LMSTUDIO_BASE_URL:-}"
  local host=""

  if [[ -n "$explicit_base" ]]; then
    printf '%s\n' "${explicit_base%/}"
    return 0
  fi

  if probe_http_base "1234" "/v1/models"; then
    return 0
  fi

  ollama_base="$(resolve_ollama_v1_base)"
  host="${ollama_base#http://}"
  host="${host%:11434}"
  printf 'http://%s:1234\n' "$host"
}

is_http_ok() {
  local url="$1"
  curl --max-time 3 -fsS "$url" >/dev/null 2>&1
}
