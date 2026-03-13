#!/usr/bin/env bash
# Run trained VizDoom agent with Sample-Factory (visualize / evaluate).
#
# Usage:
#   ./scripts/sample_factory_enjoy.sh [ENV] [EXPERIMENT]
# Examples:
#   ./scripts/sample_factory_enjoy.sh doom_basic DoomSOTA_doom_basic
#   ./scripts/sample_factory_enjoy.sh doom_defend_the_center DoomSOTA_doom_defend_the_center

set -e
ENV="${1:-doom_basic}"
EXPERIMENT="${2:-DoomSOTA_${ENV}}"
TRAIN_DIR="${TRAIN_DIR:-./train_dir}"

python -m sf_examples.vizdoom.enjoy_vizdoom \
  --env="$ENV" \
  --experiment="$EXPERIMENT" \
  --train_dir="$TRAIN_DIR" \
  "$@"
