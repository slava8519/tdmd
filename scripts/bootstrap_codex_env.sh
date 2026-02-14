#!/usr/bin/env bash
set -euo pipefail

# TDMD Codex workstation bootstrap.
# Safe to rerun.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 is required" >&2
  exit 1
fi

# System packages used by Codex workflows.
sudo apt-get update
sudo apt-get install -y \
  python3.12-venv \
  openmpi-bin libopenmpi-dev \
  ripgrep fd-find jq yq bat git-delta shellcheck hyperfine ncdu

if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'

# Convenience aliases for Ubuntu package names.
BASHRC="${HOME}/.bashrc"
BLOCK_START="# >>> tdmd-codex aliases >>>"
BLOCK_END="# <<< tdmd-codex aliases <<<"

if [ -f "$BASHRC" ]; then
  if ! grep -Fq "$BLOCK_START" "$BASHRC"; then
    {
      echo
      echo "$BLOCK_START"
      echo "alias fd='fdfind'"
      echo "alias bat='batcat'"
      echo "$BLOCK_END"
    } >> "$BASHRC"
  fi
fi

echo "Bootstrap complete."
echo "Reload shell: source ~/.bashrc"
echo "Python env: .venv/bin/python"
