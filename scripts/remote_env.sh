#!/usr/bin/env bash
# Remote connection configuration loader (no secrets stored in git).
#
# Reads scripts/remote.env (gitignored) if present. Key-based auth only;
# the password is never stored in any committed file.
#
# scripts/remote.env example:
#   export OMNIKV_HOST="root@connect.example.com"
#   export OMNIKV_PORT="22"
set -euo pipefail

export OMNIKV_HOST="${OMNIKV_HOST:-root@connect.nmb2.seetacloud.com}"
export OMNIKV_PORT="${OMNIKV_PORT:-44740}"
export OMNIKV_DIR="${OMNIKV_DIR:-/root/autodl-tmp/omnikv}"

# Optional per-user overrides (gitignored).
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/remote.env" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "${BASH_SOURCE[0]}")/remote.env"
fi
