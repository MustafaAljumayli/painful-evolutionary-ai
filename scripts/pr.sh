#!/usr/bin/env bash
# Opens a pull request for one finished feature.
#
#   ./scripts/pr.sh <name>
#
# It refuses unless ./scripts/integrate.sh reported OK for exactly this commit
# and, when the loop driver ran the feature, the loop is DONE. It pushes only
# feature/<name>. It never pushes main and never merges: the user merges the
# pull request on GitHub.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
name="${1:-}"
if [ -z "$name" ]; then
  echo "usage: ./scripts/pr.sh <name>"
  exit 2
fi
branch="feature/$name"
state="../pea-$name/reports/state.json"
ok="reports/integration/$name.ok"

git rev-parse --verify --quiet "$branch" >/dev/null || { echo "REFUSED $name  there is no branch $branch"; exit 1; }
if [ -f "$state" ] && ! grep -q '"status": "done"' "$state"; then
  echo "REFUSED $name  its loop is not DONE"
  exit 1
fi
if [ ! -f "$ok" ] || [ "$(cat "$ok")" != "$(git rev-parse "$branch")" ]; then
  echo "REFUSED $name  run ./scripts/integrate.sh $name first; it must report OK for the current commit"
  exit 1
fi

git fetch --quiet origin main
ahead="$(git rev-list --count origin/main..main)"
if [ "$ahead" -gt 0 ]; then
  echo "REFUSED $name  main has $ahead commit(s) that are not on GitHub, and the pull request would show them too."
  echo "The user must run: git push origin main"
  exit 1
fi

git push --quiet -u origin "$branch"

body="$(mktemp)"
trap 'rm -f "$body"' EXIT
if [ -f "$state" ]; then
  ./scripts/loop.sh summary "$name" >"$body"
  title="$name: $(sed -n 's/^REQUEST: //p' "$body" | cut -c1-60)"
else
  {
    echo "TICKET: $name"
    echo "COMMITS:"
    git log --format='%h %s' "main..$branch"
  } >"$body"
  title="$name"
fi
printf '\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n' >>"$body"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  url="$(gh pr view "$branch" --json url --jq .url 2>/dev/null || true)"
  if [ -z "$url" ]; then
    url="$(gh pr create --base main --head "$branch" --title "$title" --body-file "$body")"
  fi
  echo "PR $name  $url"
else
  web="$(git remote get-url origin | sed -e 's#^git@github.com:#https://github.com/#' -e 's#\.git$##')"
  echo "PUSHED $name  no pull request yet: gh is not installed or not signed in."
  echo "Open it here: $web/compare/main...$branch?expand=1"
  echo "To have it opened for you next time: brew install gh && gh auth login"
fi
