#!/usr/bin/env bash
# Starts Claude Code for one role of the loop, with the model and effort set
# in loop.conf. Solo's agent tools call this, so the configuration lives in
# the repo and not in Solo's settings:
#
#   bash scripts/agent.sh lead        the agent the user talks to
#   bash scripts/agent.sh loop        one per ticket, started by the lead
#   bash scripts/agent.sh executor    one per subtask, started by a loop
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/loop.conf"

# Use the Claude subscription login. With this variable set, Claude Code
# would bill the API account instead.
unset ANTHROPIC_API_KEY

role="${1:-}"
[ "$#" -gt 0 ] && shift
case "$role" in
  lead)
    exec claude --model "$LEAD_MODEL" --effort "$LEAD_EFFORT" "$@"
    ;;
  loop)
    exec claude --model "$LOOP_MODEL" --effort "$LOOP_EFFORT" "$@"
    ;;
  executor)
    # An executor only reads, edits and runs the checks. It loads no MCP
    # servers, so their tool definitions are not sent with every request.
    exec claude --agent executor --model "$EXECUTOR_MODEL" --effort "$EXECUTOR_EFFORT" \
      --strict-mcp-config --disallowedTools "mcp__*" "$@"
    ;;
  *)
    echo "usage: bash scripts/agent.sh lead | loop | executor" >&2
    exit 2
    ;;
esac
