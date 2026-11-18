#!/usr/bin/env bash
# Fails if this branch changes a path listed under PROTECTED_PATHS in loop.conf.
# On main it does nothing: protected paths are changed there, by a person.
# The list is read from the base branch, so a branch cannot loosen it.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ "$(git branch --show-current)" = "main" ] && exit 0
base_ref="${PEA_CHECK_BASE:-main}"
git rev-parse --verify --quiet "$base_ref" >/dev/null || exit 0

PROTECTED_PATHS=""
line="$(git show "$base_ref:loop.conf" 2>/dev/null | grep '^PROTECTED_PATHS=' || true)"
[ -n "$line" ] || line="$(grep '^PROTECTED_PATHS=' loop.conf || true)"
eval "$line"

base="$(git merge-base "$base_ref" HEAD)"
status=0
while IFS= read -r file; do
  [ -n "$file" ] || continue
  for prefix in $PROTECTED_PATHS; do
    case "$file" in
      "$prefix"*)
        echo "protected: $file"
        status=1
        ;;
    esac
  done
done < <({
  git diff --name-only "$base"
  git ls-files --others --exclude-standard
} | sort -u)

if [ "$status" -ne 0 ]; then
  echo "A feature branch may not change these. Undo the change and report what is needed instead."
fi
exit "$status"
