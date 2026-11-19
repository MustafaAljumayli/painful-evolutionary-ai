#!/usr/bin/env bash
# Starts GitHub Copilot CLI for one role of the loop, with no MCP servers.
# Solo's agent tools call it through two small commands:
#
#   scripts/copilot-lead     the agent the user talks to
#   scripts/copilot-stage    one stage of one loop, started by the lead in the
#                            loop's worktree. Which role is in reports/next.env,
#                            written by the loop driver (scripts/loop.py).
#
# The lead controls Solo with the `solo` command-line tool (see the solo
# skill), not with Solo's MCP server. To pick a model, set COPILOT_MODEL to
# an ID from Copilot's /model list before starting Solo.
set -euo pipefail

role="${1:-}"
[ "$#" -gt 0 ] && shift

# Solo checks a tool by running its command the way it would run `copilot`
# itself. Hand anything that is not the start of a session straight through.
case "${1:-}" in
  -v | --version | -h | --help | help | version | update | login | logout | mcp | plugin)
    exec copilot "$@"
    ;;
esac

common=(--disable-builtin-mcps --no-ask-user --autopilot)
[ -n "${COPILOT_MODEL:-}" ] && common+=(--model "$COPILOT_MODEL")
# What every role may do without asking: read with git, use the project's
# read-only helpers, and never push.
common+=(
  "--allow-tool=write"
  "--allow-tool=shell(./scripts/check.sh)"
  "--allow-tool=shell(./scripts/graph.sh)"
  "--allow-tool=shell(./scripts/facts.sh)"
  "--allow-tool=shell(./scripts/paper.sh)"
  "--allow-tool=shell(git status)"
  "--allow-tool=shell(git log)"
  "--allow-tool=shell(git diff)"
  "--allow-tool=shell(git show)"
  "--allow-tool=shell(git branch)"
  "--deny-tool=shell(git push)"
  "--allow-url=arxiv.org"
  "--allow-url=github.com"
  "--allow-url=raw.githubusercontent.com"
  "--allow-url=huggingface.co"
)

case "$role" in
  lead)
    exec copilot "${common[@]}" --max-autopilot-continues 500 \
      "--allow-tool=shell(solo)" \
      "--allow-tool=shell(./scripts/solo-wait.sh)" \
      "--allow-tool=shell(./scripts/loop.sh)" \
      "--allow-tool=shell(./scripts/integrate.sh)" \
      "--allow-tool=shell(./scripts/pr.sh)" \
      "--allow-tool=shell(git fetch)" \
      "--allow-tool=shell(git merge --ff-only)" \
      "$@"
    ;;
  stage)
    if [ ! -f reports/next.env ]; then
      echo "No stage is waiting in this folder (reports/next.env is missing)." >&2
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
    exec copilot "${common[@]}" --max-autopilot-continues 60 \
      --agent "$STAGE_ROLE" \
      "--allow-tool=shell(git add)" \
      "--allow-tool=shell(git commit)" \
      "--allow-tool=shell(git -C third_party/Pain-axis)" \
      "$@" -i "Read reports/ledger.md and do the stage under Now."
    ;;
  *)
    echo "usage: bash scripts/copilot-agent.sh lead | stage" >&2
    exit 2
    ;;
esac
