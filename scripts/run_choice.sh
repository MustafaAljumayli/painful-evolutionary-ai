#!/usr/bin/env bash
# Launcher for the two-button runner, for a person to start as a Solo process.
# No agent runs this.
#
#   ./scripts/run_choice.sh 8B            comprehension, then the hook-mode table run
#   BATCH_ROWS=8 ./scripts/run_choice.sh 14B
#
# Environment: BATCH_ROWS (rows per batch, default 8, an engineering choice that
# only affects speed and memory, but is logged per row and must stay fixed across
# arms and modes), MODE (hook|write, default hook), DOSE (default 1.0), and
# HF_TOKEN if the download needs one. Runs the base model (--no-adapter); the
# adapter needs peft, which environment.yml does not declare yet.
# Logs go to runs/choice_<size>_<mode>_d<dose>_<timestamp>.jsonl.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
. scripts/env.sh
if ! BIN="$(find_env_bin)"; then
  env_missing_message
  exit 1
fi

SIZE="${1:-}"
case "$SIZE" in
  8B|14B|32B) ;;
  *) echo "usage: $0 <8B|14B|32B>"; exit 2 ;;
esac

COMMON=(--size "$SIZE" --mode "${MODE:-hook}" --dose "${DOSE:-1.0}" --batch-rows "${BATCH_ROWS:-8}" --no-adapter)
"$BIN/python" -m src.run_choice comprehension "${COMMON[@]}"
"$BIN/python" -m src.run_choice run "${COMMON[@]}"
