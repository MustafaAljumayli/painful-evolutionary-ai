#!/usr/bin/env bash
# Runs each time an executor subagent tries to finish.
# While scripts/check.sh fails, this blocks the executor (exit 2) and hands it
# the output so it fixes its own work. After 3 blocks it lets the executor
# stop, and QA catches whatever is left.
set -u

MAX_BLOCKS=3
input="$(cat)"

field() {
  printf '%s' "$input" | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get(sys.argv[1]) or "")
except Exception:
    print("")
' "$1"
}

agent_id="$(field agent_id)"
cwd="$(field cwd)"
if [ -n "$cwd" ] && [ -d "$cwd" ]; then cd "$cwd" || exit 0; fi

root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
[ -x "$root/scripts/check.sh" ] || exit 0

count_file="${TMPDIR:-/tmp}/executor-stop-${agent_id:-unknown}.count"
blocks="$(cat "$count_file" 2>/dev/null || echo 0)"

if output="$("$root/scripts/check.sh" 2>&1)"; then
  rm -f "$count_file"
  exit 0
fi

if [ "$blocks" -ge "$MAX_BLOCKS" ]; then
  rm -f "$count_file"
  exit 0
fi

echo $((blocks + 1)) > "$count_file"
echo "check failed"
{
  echo "scripts/check.sh failed (try $((blocks + 1)) of $MAX_BLOCKS). Fix this, then finish:"
  printf '%s\n' "$output" | tail -n 60
} >&2
exit 2
