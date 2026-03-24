#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/output/sources"

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib-host-endpoints.sh"

# --- Security helpers ---

sanitize_stderr() {
  sed -E 's/(Authorization: Bearer )[^ ]*/\1[REDACTED]/gi; s/(token=)[^ ]*/\1[REDACTED]/gi; s/(ghp_|gho_|github_pat_)[A-Za-z0-9_]+/[REDACTED]/g'
}

validate_host() {
  local url_host="$1"
  # Strip port number
  url_host="${url_host%%:*}"
  local IFS=','
  for allowed_host in ${PRIVATE_DOCS_HOSTS:-}; do
    allowed_host="$(printf '%s' "$allowed_host" | tr -d '[:space:]')"
    if [[ "$url_host" == "$allowed_host" ]]; then
      return 0
    fi
  done
  return 1
}

topic=""
token_env=""
urls=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      topic="$2"
      shift 2
      ;;
    --token-env)
      token_env="$2"
      shift 2
      ;;
    *)
      urls+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$topic" || ${#urls[@]} -eq 0 ]]; then
  echo "Usage: ./scripts/fetch-private-gh.sh --topic <topic> [--token-env ENV] <url> [url...]" >&2
  exit 1
fi

if command -v agent-reach >/dev/null 2>&1; then
  agent-reach doctor || true
fi

topic_slug="$(printf '%s' "$topic" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')"
if [[ -z "$topic_slug" ]]; then
  topic_slug="$(printf '%s' "$topic" | md5sum | cut -c1-8)"
fi
session_id="$(date +%Y%m%d-%H%M%S)-private-${topic_slug}"
session_dir="$OUTPUT_ROOT/$session_id"
mkdir -p "$session_dir"

fetch_generic_token_url() {
  local url="$1"
  local env_name="$2"
  local token="${!env_name:-}"
  local host

  # C-3: Enforce HTTPS for token-bearing requests
  if [[ "$url" != https://* ]]; then
    echo "Refusing to send token over non-HTTPS URL: $url" >&2
    return 1
  fi

  if [[ -z "$token" ]]; then
    echo "Missing token env: $env_name" >&2
    return 1
  fi

  host="$(printf '%s' "$url" | sed -E 's#https?://([^/]+).*#\1#')"

  # C-2: Exact host match with port stripping
  if ! validate_host "$host"; then
    echo "Host is not allowlisted in PRIVATE_DOCS_HOSTS: $host" >&2
    return 1
  fi

  curl -fsSL -H "Authorization: Bearer $token" "$url"
}

fetch_github_blob() {
  local url="$1"
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI is required for GitHub private fetch." >&2
    return 1
  fi

  local stripped path owner repo ref file
  stripped="${url#https://github.com/}"
  path="${stripped#*/}"
  owner="${stripped%%/*}"
  repo="${path%%/*}"
  path="${path#*/}"

  if [[ "$path" == blob/* ]]; then
    path="${path#blob/}"
    ref="${path%%/*}"
    file="${path#*/}"

    # C-4: Validate extracted components against path traversal
    if [[ "$owner" == *..* || "$repo" == *..* || "$ref" == *..* || "$file" == *..* ]]; then
      echo "Invalid GitHub URL contains path traversal: $url" >&2
      return 1
    fi
    if [[ -z "$owner" || -z "$repo" || -z "$ref" || -z "$file" ]]; then
      echo "Incomplete GitHub URL: $url" >&2
      return 1
    fi

    local encoded_file
    encoded_file="$(jq -rn --arg f "$file" '$f|@uri')"
    gh api -H "Accept: application/vnd.github.raw" "/repos/${owner}/${repo}/contents/${encoded_file}?ref=${ref}"
    return
  fi

  if [[ "$url" == https://raw.githubusercontent.com/* ]]; then
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
      curl -fsSL -H "Authorization: Bearer ${GITHUB_TOKEN}" "$url"
    else
      curl -fsSL "$url"
    fi
    return
  fi

  echo "Unsupported GitHub URL shape: $url" >&2
  return 1
}

for url in "${urls[@]}"; do
  slug="$(printf '%s' "$url" | sed -E 's#https?://##; s#[^A-Za-z0-9._-]+#-#g; s#-+#-#g; s#(^-|-$)##g' | cut -c1-80)"
  base="$session_dir/$slug"
  status_file="${base}.status.json"
  raw_file="${base}.raw.json"
  md_file="${base}.md"
  started_at="$(date -Iseconds)"

  if [[ "$url" == https://github.com/* || "$url" == https://raw.githubusercontent.com/* ]]; then
    if result="$(fetch_github_blob "$url" 2> >(sanitize_stderr > "${base}.stderr"))"; then
      method="github"
    else
      method=""
    fi
  else
    if [[ -z "$token_env" ]]; then
      echo "Generic private docs require --token-env <ENVNAME>" >"${base}.stderr"
      method=""
    elif result="$(fetch_generic_token_url "$url" "$token_env" 2> >(sanitize_stderr >> "${base}.stderr"))"; then
      method="token-docs"
    else
      method=""
    fi
  fi

  if [[ -z "$method" ]]; then
    jq -n \
      --arg url "$url" \
      --arg topic "$topic" \
      --arg session_id "$session_id" \
      --arg started_at "$started_at" \
      --arg finished_at "$(date -Iseconds)" \
      '{url: $url, topic: $topic, session_id: $session_id, status: "failed", visibility: "private", fetch_method: null, extraction: null, started_at: $started_at, finished_at: $finished_at}' \
      > "$status_file"
    continue
  fi

  printf '%s\n' "$result" > "$raw_file"
  printf '%s\n' "$result" > "$md_file"
  jq -n \
    --arg url "$url" \
    --arg topic "$topic" \
    --arg session_id "$session_id" \
    --arg method "$method" \
    --arg started_at "$started_at" \
    --arg finished_at "$(date -Iseconds)" \
    --arg markdown_path "${md_file#$REPO_ROOT/}" \
    --arg raw_path "${raw_file#$REPO_ROOT/}" \
    '{url: $url, topic: $topic, session_id: $session_id, status: "completed", visibility: "private", fetch_method: $method, extraction: "raw", started_at: $started_at, finished_at: $finished_at, markdown_path: $markdown_path, raw_path: $raw_path}' \
    > "$status_file"
done

# Clean up empty stderr files
find "$session_dir" -name '*.stderr' -empty -delete 2>/dev/null || true

echo "$session_id"
