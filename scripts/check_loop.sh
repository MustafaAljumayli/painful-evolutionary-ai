#!/usr/bin/env bash
# Fails if an agent file sets a different model or effort from loop.conf.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
. ./loop.conf

field() { # field <file> <key>: the key's value in the file's front matter
  awk -v key="$2" '
    /^---$/ { n++; next }
    n == 1 && $1 == key ":" { print $2; exit }
  ' "$1"
}

status=0
expect() { # expect <role> <key> <wanted value>
  local file=".claude/agents/$1.md" got
  got="$(field "$file" "$2")"
  if [ "$got" != "$3" ]; then
    echo "$file: $2 is '${got:-unset}', loop.conf says '$3'"
    status=1
  fi
}

expect researcher model "$RESEARCHER_MODEL"
expect researcher effort "$RESEARCHER_EFFORT"
expect planner model "$PLANNER_MODEL"
expect planner effort "$PLANNER_EFFORT"
expect executor model "$EXECUTOR_MODEL"
expect executor effort "$EXECUTOR_EFFORT"
# QA's file carries the stronger model, so a run that forgets to choose one
# errs on the careful side.
expect qa model "$QA_MODEL_CRITICAL"
expect qa effort "$QA_EFFORT"

if [ "$status" -ne 0 ]; then
  echo "Change loop.conf and the agent files together (.claude/agents/ and .github/agents/)."
fi
exit "$status"
