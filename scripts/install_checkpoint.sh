#!/usr/bin/env bash
# Fetch the released checkpoint into a Sample Factory train_dir layout.
#
# Usage:
#   ./scripts/install_checkpoint.sh                 # -> ./train_dir/gru_250m_baseline
#   TRAIN_DIR=/path/to/train_dir ./scripts/install_checkpoint.sh
#   RELEASE=v1-gru-250m EXPERIMENT=gru_250m_baseline ./scripts/install_checkpoint.sh
#
# Needs either the `gh` CLI (authenticated, for private forks) or `curl`. The
# release holds an ordinary Sample Factory experiment directory:
#   config.json + checkpoint_p0/{best,checkpoint}_000015260_250019840*.pth
set -euo pipefail

REPO="${REPO:-Kiraprint/DOOM-Mamba2-agent}"
RELEASE="${RELEASE:-v1-gru-250m}"
EXPERIMENT="${EXPERIMENT:-gru_250m_baseline}"
TRAIN_DIR="${TRAIN_DIR:-./train_dir}"
DEST="$TRAIN_DIR/$EXPERIMENT/checkpoint_p0"

base="https://github.com/$REPO/releases/download/$RELEASE"
mkdir -p "$DEST"

fetch() {
  local name="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "have $out"
    return
  fi
  echo "fetching $name"
  if command -v gh >/dev/null 2>&1; then
    gh release download "$RELEASE" --repo "$REPO" --pattern "$name" --dir "$(dirname "$out")" --clobber
  else
    curl -fL --retry 3 -o "$out" "$base/$name"
  fi
}

fetch config.json "$TRAIN_DIR/$EXPERIMENT/config.json"
fetch best_000015260_250019840_reward_24.480.pth "$DEST/best_000015260_250019840_reward_24.480.pth"
fetch checkpoint_000015260_250019840.pth "$DEST/checkpoint_000015260_250019840.pth"

echo
echo "installed into $TRAIN_DIR/$EXPERIMENT"
ls -l "$TRAIN_DIR/$EXPERIMENT/config.json" "$DEST"
echo
echo "next: EXPERIMENT=$EXPERIMENT ./scripts/eval_sf_checkpoint.sh"
