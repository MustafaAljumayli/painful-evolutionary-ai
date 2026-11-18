#!/usr/bin/env bash
# The loop driver, run from the project environment. See scripts/loop.py.
#   ./scripts/loop.sh start <name> | next <name> | answer <name> | summary <name> | status
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/scripts/env.sh"
if ! BIN="$(cd "$here" && find_env_bin)"; then
  env_missing_message
  exit 1
fi
exec "$BIN/python" "$here/scripts/loop.py" "$@"
