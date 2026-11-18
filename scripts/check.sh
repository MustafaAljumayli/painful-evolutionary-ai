#!/usr/bin/env bash
# The one definition of "it compiles and meets the standards".
# The executor stop hook, the QA agent and you all run this same script.
# Exit 0 = pass. Anything else = fail, with the reasons printed.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

# CLAUDE.md and AGENTS.md carry the same project context under two names.
# They must stay identical so every tool and every agent reads the same thing.
if ! cmp -s CLAUDE.md AGENTS.md; then
  echo "AGENTS.md is out of sync with CLAUDE.md. Run: cp CLAUDE.md AGENTS.md"
  exit 1
fi

# A worktree is a fresh checkout with no .venv of its own, so fall back to the
# main checkout's .venv, then to whatever python3 is on PATH.
main_root="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
if [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
elif [ -x "$main_root/.venv/bin/python" ]; then
  PY="$main_root/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi
if [ -z "$PY" ]; then
  echo "No Python found. Create a venv: python3 -m venv .venv"
  exit 1
fi

targets=()
for d in src tests; do
  [ -d "$d" ] && targets+=("$d")
done
if [ "${#targets[@]}" -eq 0 ]; then
  echo "Nothing to check yet: no src/ or tests/ folder."
  exit 0
fi

for tool in ruff pytest; do
  if ! "$PY" -m "$tool" --version >/dev/null 2>&1; then
    echo "$tool is not installed for $PY. Run: $PY -m pip install ruff pytest"
    exit 1
  fi
done

fail=0

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
