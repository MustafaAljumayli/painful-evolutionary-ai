#!/usr/bin/env bash
# Finds the project's conda environment (declared in environment.yml) without
# needing it to be activated. Source this file, then call find_env_bin.
# A worktree needs no setup of its own: the environment is shared by name.

PEA_ENV_NAME="painful-evolutionary-ai"

# Prints the environment's bin directory, or returns 1 if it does not exist.
# Set PEA_ENV_BIN to point at a different environment's bin directory.
find_env_bin() {
  if [ -n "${PEA_ENV_BIN:-}" ]; then
    if [ -x "$PEA_ENV_BIN/python" ]; then
      echo "$PEA_ENV_BIN"
      return 0
    fi
    return 1
  fi
  if [ "${CONDA_DEFAULT_ENV:-}" = "$PEA_ENV_NAME" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
    echo "$CONDA_PREFIX/bin"
    return 0
  fi
  local bases=()
  if [ -n "${CONDA_EXE:-}" ]; then
    bases+=("$(dirname "$(dirname "$CONDA_EXE")")")
  fi
  bases+=(
    "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3"
    "/opt/homebrew/Caskroom/miniconda/base" "/opt/miniconda3" "/opt/anaconda3"
  )
  local base
  for base in "${bases[@]}"; do
    if [ -x "$base/envs/$PEA_ENV_NAME/bin/python" ]; then
      echo "$base/envs/$PEA_ENV_NAME/bin"
      return 0
    fi
  done
  return 1
}

env_missing_message() {
  echo "The conda environment '$PEA_ENV_NAME' was not found."
  echo "Create it from the main checkout:  conda env create -f environment.yml"
}
