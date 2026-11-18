#!/usr/bin/env bash
# Copilot CLI version of the executor stop check.
# Runs each time an executor agent tries to finish. While scripts/check.sh
# fails, it prints {"decision":"block","reason":...} so Copilot sends the
# executor back to fix its own work. After 3 blocks it lets the executor stop,
# and QA catches whatever is left.
set -u

MAX_BLOCKS=3
input="$(cat)"

field() {
  printf '%s' "$input" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
for key in sys.argv[1:]:
    if data.get(key):
        print(data[key])
        break
' "$@"
}

agent_id="$(field agentId agent_id sessionId session_id)"
cwd="$(field cwd)"
if [ -n "$cwd" ] && [ -d "$cwd" ]; then cd "$cwd" || exit 0; fi

root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
[ -f "$root/scripts/check.sh" ] || exit 0

count_file="${TMPDIR:-/tmp}/copilot-executor-stop-${agent_id:-unknown}.count"
blocks="$(cat "$count_file" 2>/dev/null || echo 0)"

if output="$(bash "$root/scripts/check.sh" 2>&1)"; then
  rm -f "$count_file"
  exit 0
fi

if [ "$blocks" -ge "$MAX_BLOCKS" ]; then
  rm -f "$count_file"
  exit 0
fi

echo $((blocks + 1)) > "$count_file"
{
  echo "scripts/check.sh failed (try $((blocks + 1)) of $MAX_BLOCKS). Fix this, then finish:"
  printf '%s\n' "$output" | tail -n 60
} | python3 -c '
import json, sys
print(json.dumps({"decision": "block", "reason": sys.stdin.read()}))
'
exit 0
