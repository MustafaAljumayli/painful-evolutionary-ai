#!/usr/bin/env bash
# Prints the model QA should run on for this branch: the stronger one if the
# branch changes any path under CRITICAL_PATHS in loop.conf, the cheaper one
# otherwise.
#   ./scripts/qa_model.sh [base branch, default main]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
. ./loop.conf
base="${1:-main}"
while IFS= read -r file; do
  for prefix in $CRITICAL_PATHS; do
    case "$file" in
      "$prefix"*)
        echo "$QA_MODEL_CRITICAL"
        exit 0
        ;;
    esac
  done
done < <(git diff --name-only "$base"...HEAD)
echo "$QA_MODEL_PLUMBING"
