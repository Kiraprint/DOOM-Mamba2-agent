#!/usr/bin/env bash
# Train VizDoom agent with Sample-Factory (SOTA reproduction).
# See https://www.samplefactory.dev/09-environment-integrations/vizdoom/
#
# Usage:
#   ./scripts/sample_factory_train.sh [ENV] [STEPS]
# Examples:
#   ./scripts/sample_factory_train.sh doom_basic 2000000
#   ./scripts/sample_factory_train.sh doom_defend_the_center 4000000000
#
# Paper envs: doom_basic, doom_deadly_corridor, doom_defend_the_center,
#             doom_defend_the_line, doom_health_gathering, doom_health_gathering_supreme,
#             doom_battle, doom_battle2

set -e
ENV="${1:-doom_basic}"
STEPS="${2:-4000000000}"
EXPERIMENT="DoomSOTA_${ENV}"
TRAIN_DIR="${TRAIN_DIR:-./train_dir}"

# ~10-core machine: 20 workers × 16 envs; scale down for fewer cores
NUM_WORKERS="${NUM_WORKERS:-20}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-16}"

python -m sf_examples.vizdoom.train_vizdoom \
  --env="$ENV" \
  --experiment="$EXPERIMENT" \
  --train_dir="$TRAIN_DIR" \
  --train_for_env_steps="$STEPS" \
  --algo=APPO \
  --use_rnn=True \
  --env_frameskip=4 \
  --num_workers="$NUM_WORKERS" \
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER" \
  --res_w=128 \
  --res_h=72 \
  "$@"
