#!/usr/bin/env bash
# Evaluate a released checkpoint with Sample Factory, optionally recording video.
#
# Usage:
#   ./scripts/eval_sf_checkpoint.sh                        # doom_benchmark, 5 episodes
#   EXPERIMENT=gru_250m_baseline ./scripts/eval_sf_checkpoint.sh
#   MAX_EPISODES=3 SAVE_VIDEO=1 ./scripts/eval_sf_checkpoint.sh
#
# Env vars: EXPERIMENT, TRAIN_DIR, MAX_EPISODES, SAVE_VIDEO, VIDEO_FRAMES,
#           VIDEO_NAME, NO_RENDER, EXTRA (appended verbatim to the command).
#
# The flags below are the ones recorded in train_dir/<experiment>/config.json of
# the run being evaluated -- env, algo, frameskip, resolution, rnn size, batch,
# rollout, recurrence, gamma. Change nothing here without saying so in the
# results, or the number stops being comparable with results/CLUSTER_RESULTS.md.
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-gru_250m_baseline}"
TRAIN_DIR="${TRAIN_DIR:-./train_dir}"
MAX_EPISODES="${MAX_EPISODES:-5}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_FRAMES="${VIDEO_FRAMES:-2000}"
VIDEO_NAME="${VIDEO_NAME:-${EXPERIMENT}_rollout}"
NO_RENDER="${NO_RENDER:-0}"

if [[ ! -d "$TRAIN_DIR/$EXPERIMENT" ]]; then
  echo "no such experiment: $TRAIN_DIR/$EXPERIMENT" >&2
  echo "download the released checkpoint first (see README, 'Released checkpoint')." >&2
  exit 1
fi

ARGS=(
  --env=doom_benchmark
  --experiment="$EXPERIMENT"
  --train_dir="$TRAIN_DIR"
  --algo=APPO
  --use_rnn=True
  --env_frameskip=4
  --res_w=128
  --res_h=72
  --rnn_size=512
  --rnn_num_layers=1
  --batch_size=4096
  --rollout=64
  --recurrence=32
  --gamma=0.99
  --max_num_episodes="$MAX_EPISODES"
)

if [[ "$SAVE_VIDEO" == "1" ]]; then
  ARGS+=(
    --save_video
    --video_frames="$VIDEO_FRAMES"
    --video_name="$VIDEO_NAME"
    --fps=35
  )
else
  ARGS+=(--fps=35)
fi
if [[ "$NO_RENDER" == "1" ]]; then
  ARGS+=(--no_render)
fi

# EXTRA is intentionally word-split so callers can add flags.
# shellcheck disable=SC2206
if [[ -n "${EXTRA:-}" ]]; then
  ARGS+=($EXTRA)
fi

echo "python -m sf_examples.vizdoom.enjoy_vizdoom ${ARGS[*]}"
exec python -m sf_examples.vizdoom.enjoy_vizdoom "${ARGS[@]}"
