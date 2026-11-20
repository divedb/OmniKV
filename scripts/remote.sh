#!/usr/bin/env bash
# Run a command inside the remote OmniKV project directory.
#
# Usage: scripts/remote.sh <command...>
#   scripts/remote.sh 'nvidia-smi'
#   scripts/remote.sh '.venv/bin/python -c "import torch; print(torch.cuda.is_available())"'
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/remote_env.sh

exec ssh -p "$OMNIKV_PORT" "$OMNIKV_HOST" "cd $OMNIKV_DIR && $*"
