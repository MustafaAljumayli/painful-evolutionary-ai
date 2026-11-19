#!/usr/bin/env bash
# Waits until one of several Solo processes goes quiet or exits. This stands
# in for Solo's idle timers, which exist only in its MCP server; it uses the
# `solo` command-line tool instead.
#
#   ./scripts/solo-wait.sh [--project-id <id>] <process id> [<process id> ...]
#
# Prints "IDLE <id>" and the end of that process's output when its output has
# not changed for SOLO_WAIT_QUIET seconds (default 30), or "EXITED <id>" when
# it has stopped. After SOLO_WAIT_TIMEOUT seconds (default 240) it prints
# "STILL WORKING <ids>" and returns, so the caller can run it again.
set -uo pipefail
quiet="${SOLO_WAIT_QUIET:-30}"
timeout="${SOLO_WAIT_TIMEOUT:-240}"
poll="${SOLO_WAIT_POLL:-5}"
project=()
if [ "${1:-}" = "--project-id" ]; then
  project=(--project-id "$2")
  shift 2
fi
if [ "$#" -lt 1 ]; then
  echo "usage: ./scripts/solo-wait.sh [--project-id <id>] <process id> [<process id> ...]"
  exit 2
fi
if ! command -v solo >/dev/null 2>&1; then
  echo "The solo command is not installed. In Solo: Settings > Integrations > HTTP API & CLI."
  exit 2
fi

ids=("$@")
last=()
since=()
start="$(date +%s)"
for i in "${!ids[@]}"; do
  last[i]=""
  since[i]="$start"
done

while :; do
  now="$(date +%s)"
  for i in "${!ids[@]}"; do
    id="${ids[i]}"
    status="$(solo processes get "$id" --json 2>/dev/null | tr -d ' \n' | sed -n 's/.*"status":"\([a-z_]*\)".*/\1/p')"
    case "$status" in
      exited | stopped | failed)
        echo "EXITED $id  (status: $status)"
        solo processes output "$id" ${project[@]+"${project[@]}"} --lines 20 2>/dev/null
        exit 0
        ;;
    esac
    text="$(solo processes output "$id" ${project[@]+"${project[@]}"} --lines 40 2>/dev/null)"
    sum="$(printf '%s' "$text" | cksum)"
    if [ "$sum" != "${last[i]}" ]; then
      last[i]="$sum"
      since[i]="$now"
    elif [ $((now - since[i])) -ge "$quiet" ]; then
      echo "IDLE $id  (no new output for ${quiet}s)"
      printf '%s\n' "$text" | tail -n 20
      exit 0
    fi
  done
  if [ $((now - start)) -ge "$timeout" ]; then
    echo "STILL WORKING ${ids[*]}  (run this again)"
    exit 0
  fi
  sleep "$poll"
done
