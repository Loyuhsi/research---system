#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/output/sources"

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib-host-endpoints.sh"

if command -v mcporter >/dev/null 2>&1; then
  MCPORTER=(mcporter)
else
  MCPORTER=(npx mcporter)
fi

topic=""
urls=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      topic="$2"
      shift 2
      ;;
    *)
      urls+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$topic" || ${#urls[@]} -eq 0 ]]; then
  echo "Usage: ./scripts/fetch-public.sh --topic <topic> <url> [url...]" >&2
  exit 1
fi

topic_slug="$(printf '%s' "$topic" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')"
if [[ -z "$topic_slug" ]]; then
  topic_slug="$(printf '%s' "$topic" | md5sum | cut -c1-8)"
fi
session_id="$(date +%Y%m%d-%H%M%S)-${topic_slug}"
session_dir="$OUTPUT_ROOT/$session_id"
mkdir -p "$session_dir"

extract_content() {
  jq -r '
    def unwrap: if type == "array" then map(select(. != "" and . != null)) | join("\n\n") else . end;
    (.markdown // .content // .text // .result.markdown // .result.content // .result.text // .data.markdown // .data.content // .data.text // empty) | unwrap
  ' 2>/dev/null
}

fetch_once() {
  local expression="$1"
  "${MCPORTER[@]}" --config "$REPO_ROOT/config/mcporter.json" call "$expression" --output json
}

for url in "${urls[@]}"; do
  slug="$(printf '%s' "$url" | sed -E 's#https?://##; s#[^A-Za-z0-9._-]+#-#g; s#-+#-#g; s#(^-|-$)##g' | cut -c1-80)"
  base="$session_dir/$slug"
  status_file="${base}.status.json"
  raw_file="${base}.raw.json"
  md_file="${base}.md"
  started_at="$(date -Iseconds)"

  if result="$(retry_cmd fetch_once "scrapling.get(url: \"$url\", extraction_type: \"markdown\")" 2>"${base}.stderr")"; then
    method="scrapling.get"
  elif result="$(retry_cmd fetch_once "scrapling.fetch(url: \"$url\", headless: true)" 2>>"${base}.stderr")"; then
    method="scrapling.fetch"
  else
    jq -n \
      --arg url "$url" \
      --arg topic "$topic" \
      --arg session_id "$session_id" \
      --arg started_at "$started_at" \
      --arg finished_at "$(date -Iseconds)" \
      '{url: $url, topic: $topic, session_id: $session_id, status: "failed", visibility: "public", fetch_method: null, extraction: null, started_at: $started_at, finished_at: $finished_at}' \
      > "$status_file"
    continue
  fi

  printf '%s\n' "$result" > "$raw_file"
  extraction="extracted"
  content="$(printf '%s' "$result" | extract_content || true)"
  if [[ -z "$content" ]]; then
    echo "[warn] Could not extract content from response for $url; using raw response" >&2
    content="$(printf '%s\n' "$result")"
    extraction="raw_fallback"
  fi
  printf '%s\n' "$content" > "$md_file"

  jq -n \
    --arg url "$url" \
    --arg topic "$topic" \
    --arg session_id "$session_id" \
    --arg method "$method" \
    --arg extraction "$extraction" \
    --arg started_at "$started_at" \
    --arg finished_at "$(date -Iseconds)" \
    --arg markdown_path "${md_file#$REPO_ROOT/}" \
    --arg raw_path "${raw_file#$REPO_ROOT/}" \
    '{url: $url, topic: $topic, session_id: $session_id, status: "completed", visibility: "public", fetch_method: $method, extraction: $extraction, started_at: $started_at, finished_at: $finished_at, markdown_path: $markdown_path, raw_path: $raw_path}' \
    > "$status_file"
done

echo "$session_id"
