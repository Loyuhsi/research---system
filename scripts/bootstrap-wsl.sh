#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${HOME}/.venvs/auto-research-scrapling"
PI_DIR="${HOME}/.pi/agent"
VAULT_ROOT="${VAULT_ROOT:-/mnt/c/Users/User/Documents/AutoResearchVault}"

cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap-wsl.sh must be run inside WSL/Linux." >&2
  exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"

ensure_command() {
  local name="$1"
  local install_cmd="$2"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "[install] $name"
    eval "$install_cmd"
  fi
}

sudo apt-get update
sudo apt-get install -y curl git jq gh python3 python3-pip python3-venv pipx

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

pipx ensurepath
export PATH="${HOME}/.local/bin:${PATH}"

ensure_command pi "npm install -g @mariozechner/pi-coding-agent"
ensure_command mcporter "npm install -g mcporter"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"
if ! scrapling install; then
  echo "[warn] 'scrapling install' failed. Scrapling browser engine may not work." >&2
  echo "[warn] You may need to run 'scrapling install' manually inside the venv." >&2
fi
deactivate

if ! command -v agent-reach >/dev/null 2>&1; then
  pipx install "https://github.com/Panniantong/agent-reach/archive/main.zip"
fi

mkdir -p "$PI_DIR" "$REPO_ROOT/output/sources" "$REPO_ROOT/output/research" "$REPO_ROOT/output/notes" "$REPO_ROOT/output/logs" "$REPO_ROOT/knowledge/index" "$REPO_ROOT/knowledge/logs" "$REPO_ROOT/knowledge/memory-records" "$REPO_ROOT/knowledge/reports" "$REPO_ROOT/knowledge/datasets" "$REPO_ROOT/knowledge/skill-docs" "$REPO_ROOT/staging/tooling" "$REPO_ROOT/staging/memory-drafts" "$REPO_ROOT/staging/skills-candidates" "$REPO_ROOT/sandbox/rd-agent/in" "$REPO_ROOT/sandbox/rd-agent/out" "$REPO_ROOT/.github/skills" "$REPO_ROOT/.runtime"
mkdir -p "$VAULT_ROOT/00_Inbox" "$VAULT_ROOT/10_Research/AutoResearch" "$VAULT_ROOT/90_Logs"
chmod +x "$REPO_ROOT"/scripts/*.sh

cat > "$PI_DIR/models.json" <<'EOF'
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
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
      "baseUrl": "http://localhost:1234/v1",
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

cat > "$PI_DIR/settings.json" <<'EOF'
{
  "defaultProvider": "ollama",
  "defaultModel": "qwen3.5:9b"
}
EOF

echo "[ok] bootstrap complete"
echo "Repo root: $REPO_ROOT"
echo "Pi config: $PI_DIR"
