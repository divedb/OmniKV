#!/usr/bin/env bash
# Sync the OmniKV workspace to the remote GPU server.
#
# Local workspace is the single source of truth; the remote copy is a test
# deployment. Excludes virtualenvs, caches, models, logs and build artifacts.
#
# Usage: scripts/sync_remote.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/remote_env.sh

RSYNC_FLAGS=(-avz --delete
  --exclude '.venv/' --exclude '.git/' --exclude '__pycache__/'
  --exclude '.mypy_cache/' --exclude '.ruff_cache/' --exclude '.pytest_cache/'
  --exclude 'models/' --exclude '*.log' --exclude 'logs/'
  --exclude 'dist/' --exclude 'build/' --exclude '*.so'
  --exclude '.DS_Store'
  -e "ssh -p $OMNIKV_PORT")

if [[ "${1:-}" == "--dry-run" ]]; then
  RSYNC_FLAGS+=(-n)
fi

echo "Syncing -> $OMNIKV_HOST:/root/autodl-tmp/omnikv"
rsync "${RSYNC_FLAGS[@]}" ./ "$OMNIKV_HOST:/root/autodl-tmp/omnikv/"
