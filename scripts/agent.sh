#!/usr/bin/env bash
# Starts Claude Code for one role of the loop, with the model and effort set
# in loop.conf. Solo's agent tools call it through two small commands, so
# the configuration lives in the repo and not in Solo's settings:
#
#   scripts/claude-lead     the agent the user talks to
#   scripts/claude-stage    one stage of one loop: a researcher, planner,
#                           executor, QA or reviewer agent, started by the lead in the
#                           loop's worktree. Which one is in reports/next.env,
#                           written by the loop driver (scripts/loop.py).
#
# Each is `bash scripts/agent.sh <role>` and otherwise behaves like `claude`.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/loop.conf"

# Use the Claude subscription login. With this variable set, Claude Code
# would bill the API account instead.
unset ANTHROPIC_API_KEY

role="${1:-}"
[ "$#" -gt 0 ] && shift

# Solo checks a tool by running its command the way it would run `claude`
# itself (`--version`, `mcp ...`). Hand anything that is not the start of a
# session straight to claude, unchanged.
case "${1:-}" in
  -v | --version | -h | --help | mcp | doctor | update | upgrade | install | auth | config | plugin | plugins | agents | setup-token)
    exec claude "$@"
    ;;
esac

case "$role" in
  lead)
    exec claude --model "$LEAD_MODEL" --effort "$LEAD_EFFORT" "$@"
    ;;
  stage)
    if [ ! -f reports/next.env ]; then
      echo "No stage is waiting in this folder (reports/next.env is missing)." >&2
      echo "The lead starts stages; see ./scripts/loop.sh." >&2
      exit 1
    fi
    . reports/next.env
    case "$STAGE_ROLE" in
      researcher | planner | executor | qa | reviewer) ;;
      *)
        echo "reports/next.env names an unknown role: $STAGE_ROLE" >&2
        exit 1
        ;;
    esac
    # A stage agent only reads, edits and runs the checks. It loads no MCP
    # servers, so their tool definitions are not sent with every request. It
    # starts on its own: the ledger tells it what to do.
    exec claude --agent "$STAGE_ROLE" --model "$STAGE_MODEL" --effort "$STAGE_EFFORT" \
      --strict-mcp-config "$@" "Read reports/ledger.md and do the stage under Now."
    ;;
  *)
    echo "usage: bash scripts/agent.sh lead | stage" >&2
    exit 2
    ;;
esac
