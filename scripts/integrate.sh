#!/usr/bin/env bash
# Integration check: do the finished branches still pass together?
#
#   ./scripts/integrate.sh <name> [<name> ...]
#
# Merges feature/<name> one at a time, in the order given, into a scratch
# worktree of main (../pea-integration) and runs ./scripts/check.sh after each
# merge. Prints one line per branch:
#
#   OK <name>        passes together with main and every OK branch before it
#   BROKEN <name>    conflicts or fails the checks; it is left out and the
#                    remaining branches are tried without it
#
# Nothing is merged into main and nothing is pushed. A pull request can only
# be opened for a branch that is OK (scripts/pr.sh checks).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
if [ "$(git branch --show-current)" != "main" ] || [ "$#" -lt 1 ]; then
  echo "usage (from the main checkout): ./scripts/integrate.sh <name> [<name> ...]"
  exit 2
fi

scratch="$(dirname "$PWD")/pea-integration"
git worktree remove --force "$scratch" >/dev/null 2>&1
rm -rf "$scratch"
git worktree prune
if ! git worktree add --detach "$scratch" main >/dev/null 2>&1; then
  echo "cannot create the scratch worktree $scratch"
  exit 2
fi
# Borrow the authors' repo from this checkout instead of downloading it again.
git -C "$scratch" submodule update --init --reference "$PWD/third_party/Pain-axis" >/dev/null 2>&1 ||
  git -C "$scratch" submodule update --init >/dev/null 2>&1
mkdir -p reports/integration

merged="main"
for name in "$@"; do
  branch="feature/$name"
  rm -f "reports/integration/$name.ok"
  if ! git rev-parse --verify --quiet "$branch" >/dev/null; then
    echo "BROKEN $name  there is no branch $branch"
    continue
  fi
  if ! git -C "$scratch" merge --no-ff --no-edit "$branch" >/dev/null 2>&1; then
    files="$(git -C "$scratch" diff --name-only --diff-filter=U | tr '\n' ' ')"
    git -C "$scratch" merge --abort >/dev/null 2>&1
    echo "BROKEN $name  merge conflict with [$merged] in: $files"
    continue
  fi
  log="reports/integration/$name.log"
  if (cd "$scratch" && PEA_CHECK_BASE=HEAD ./scripts/check.sh) >"$log" 2>&1; then
    git rev-parse "$branch" >"reports/integration/$name.ok"
    echo "OK $name"
    merged="$merged + $name"
  else
    echo "BROKEN $name  the checks fail when it is merged with [$merged]. End of $log:"
    tail -n 20 "$log" | sed 's/^/    /'
    git -C "$scratch" reset --hard --quiet HEAD~1
  fi
done

git worktree remove --force "$scratch" >/dev/null 2>&1
exit 0
