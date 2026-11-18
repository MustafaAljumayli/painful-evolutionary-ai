#!/usr/bin/env bash
# The one definition of "it compiles and meets the standards".
# The executor stop hook, the QA agent, the lead and you all run this script.
# Exit 0 = pass. Anything else = fail, with the reasons printed.
#
# Everything runs from the conda environment declared in environment.yml.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

# CLAUDE.md and AGENTS.md carry the same project context under two names.
# They must stay identical so every tool and every agent reads the same thing.
if ! cmp -s CLAUDE.md AGENTS.md; then
  echo "AGENTS.md is out of sync with CLAUDE.md. Run: cp CLAUDE.md AGENTS.md"
  exit 1
fi

. scripts/env.sh
if ! BIN="$(find_env_bin)"; then
  env_missing_message
  exit 1
fi
PY="$BIN/python"

for tool in ruff pytest; do
  if ! "$PY" -m "$tool" --version >/dev/null 2>&1; then
    echo "$tool is missing from the environment. Run: conda env update -f environment.yml --prune"
    exit 1
  fi
done

fail=0

echo "== dependencies"
"$PY" scripts/check_deps.py || fail=1

echo "== paper text"
./scripts/paper.sh check || fail=1

targets=()
for d in src tests scripts; do
  [ -d "$d" ] && targets+=("$d")
done

echo "== compile"
"$PY" -m compileall -q "${targets[@]}" || fail=1

echo "== lint"
"$PY" -m ruff check . || fail=1

echo "== format"
"$PY" -m ruff format --check . || fail=1

if [ -d tests ]; then
  echo "== tests (GPU tests skipped)"
  "$PY" -m pytest -q -m "not gpu"
  rc=$?
  # pytest exits 5 when it collects no tests, which is fine on a new repo
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then fail=1; fi
fi

if [ "$fail" -ne 0 ]; then
  echo "== CHECK FAILED"
  exit 1
fi
echo "== CHECK PASSED"
