#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CODEX_BIN="${CODEX_BIN:-codex}"
SANDBOX_MODE="${CODEX_SANDBOX_MODE:-danger-full-access}"
APPROVAL_MODE="${CODEX_APPROVAL_MODE:-never}"

exec "$CODEX_BIN" -C "$ROOT_DIR" -s "$SANDBOX_MODE" -a "$APPROVAL_MODE" "$@"
