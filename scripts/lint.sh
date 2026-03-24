#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v shellcheck >/dev/null 2>&1; then
  echo "[error] shellcheck is not installed. Install it with: sudo apt-get install shellcheck" >&2
  exit 1
fi

echo "Running shellcheck on all shell scripts..."
shellcheck "$SCRIPT_DIR"/*.sh
echo "[ok] All shell scripts pass shellcheck"
