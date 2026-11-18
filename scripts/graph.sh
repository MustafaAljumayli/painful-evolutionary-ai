#!/usr/bin/env bash
# A map of the code: ours and the authors' 300-plus scripts. Ask it where
# something lives before grepping or opening files. It is built locally from
# the syntax tree, with no model calls, by graphify from the project
# environment (environment.yml).
#
#   ./scripts/graph.sh build                       build or refresh the map
#   ./scripts/graph.sh query "steering hook"      files and lines for a topic
#   ./scripts/graph.sh explain "path::symbol"     one symbol and its neighbours
#   ./scripts/graph.sh path "A" "B"               how two symbols connect
#   ./scripts/graph.sh affected "symbol"          what depends on a symbol
#
# The map lives in graphify-out/ and is not committed. Each checkout builds
# its own on first use, in under a minute, and refreshes itself when a
# Python file under src/, tests/ or scripts/ has changed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
. scripts/env.sh
if ! BIN="$(find_env_bin)"; then
  env_missing_message
  exit 1
fi
if [ ! -x "$BIN/graphify" ]; then
  echo "graphify is missing from the environment. Run: conda env update -f environment.yml --prune"
  exit 1
fi
GRAPH="graphify-out/graph.json"

build() {
  local log
  log="$("$BIN/graphify" extract . --code-only 2>&1)" || { printf '%s\n' "$log" | tail -n 5; exit 1; }
  printf '%s\n' "$log" | grep -E "wrote .*graph.json" | sed 's/^\[graphify extract\] //' || echo "graph built"
}

cmd="${1:-}"
case "$cmd" in
  build)
    build
    ;;
  query|explain|path|affected)
    shift
    # Rebuild when the map is missing or our own code changed since it was
    # built. Unchanged files come from a cache, so this takes a few seconds.
    if [ ! -f "$GRAPH" ] || [ -n "$(find src tests scripts -name '*.py' -newer "$GRAPH" 2>/dev/null | head -n 1)" ]; then
      build >/dev/null
    fi
    "$BIN/graphify" "$cmd" "$@" --graph "$GRAPH"
    ;;
  *)
    echo "usage: ./scripts/graph.sh build | query \"...\" | explain \"...\" | path \"A\" \"B\" | affected \"...\""
    exit 2
    ;;
esac
