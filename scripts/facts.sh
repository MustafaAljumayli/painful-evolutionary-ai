#!/usr/bin/env bash
# The project's research memory. Prints what earlier research briefs already
# established, so a researcher starts from it instead of re-reading sources.
#
#   ./scripts/facts.sh hook layer     lines containing every word given
#   ./scripts/facts.sh                everything (long; prefer words)
#
# It lists every [READ] finding, expected value and trap from the briefs
# under research/, each with the brief and line it came from. Nothing is
# stored twice: the briefs are the record, so this cannot go stale.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
shopt -s nullglob
briefs=(research/*.md)
if [ "${#briefs[@]}" -eq 0 ]; then
  echo "No research briefs yet."
  exit 0
fi
out="$(awk '
  FNR == 1 { section = "" }
  /^## / { section = substr($0, 4); next }
  /^- \[READ\]/ { printf "%s:%d: %s\n", FILENAME, FNR, substr($0, 3); next }
  /^- / && (section == "Traps" || section == "Expected values") {
    if ($0 == "- none") next
    printf "%s:%d: [%s] %s\n", FILENAME, FNR, toupper(section), substr($0, 3)
  }
' "${briefs[@]}")"
for word in "$@"; do
  out="$(printf '%s\n' "$out" | grep -i -F -- "$word" || true)"
done
if [ -z "$out" ]; then
  echo "Nothing established yet for: $*"
  exit 0
fi
printf '%s\n' "$out"
