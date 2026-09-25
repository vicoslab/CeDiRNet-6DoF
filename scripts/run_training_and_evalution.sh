#!/usr/bin/env bash
set -euo pipefail

# Public train/eval helper for CeDiRNet-6DoF.
# Usage:
#   CLOTH_COMPETITION_ICRA2024_DATASET=/path/to/datasets ./scripts/run_training_and_evalution.sh dry_run
#   ./scripts/run_training_and_evalution.sh train
#   ./scripts/run_training_and_evalution.sh eval
#
# Released variants default to: crop_rgb, nocrop_rgb, crop_rgbd.
# Override with VARIANTS="crop_rgb crop_rgbd", or use DEPTH_ENCODINGS/CROP_MODES for custom sweeps.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODE=${1:-dry_run}
if [[ "$MODE" != "dry_run" && "$MODE" != "train" && "$MODE" != "eval" && "$MODE" != "train_eval" ]]; then
  echo "Usage: $0 [dry_run|train|eval|train_eval]" >&2
  exit 2
fi

ensure_model() (
  local filename="$1"
  local url="$2"
  local target="$PRETRAINED_MODEL_DIR/$filename"
  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" && -s "$target" ]]; then return 0; fi
  echo "[download] $filename"
  if [[ "$url" == TODO* || -z "$url" ]]; then
    echo "Model URL for $filename is not configured yet; place it manually or edit this script." >&2
    exit 1
  fi
  # Stage beside the final file so promotion is atomic even on mounted caches.
  # The subshell keeps cleanup traps local to this one download.
  local temporary
  temporary=$(mktemp "${target}.tmp.XXXXXX")
  trap 'rm -f -- "$temporary"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  if command -v wget >/dev/null 2>&1; then
    wget -O "$temporary" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --show-error "$url" --output "$temporary"
  else
    echo "Error: neither wget nor curl is available to download $filename" >&2
    exit 1
  fi
  if [[ ! -s "$temporary" ]]; then
    echo "Error: downloaded model $filename is empty" >&2
    exit 1
  fi
  mv -f -- "$temporary" "$target"
)

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CLOTH_COMPETITION_ICRA2024_DATASET="${CLOTH_COMPETITION_ICRA2024_DATASET:-$REPO_ROOT/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/exp}"
export PRETRAINED_MODEL_DIR="${PRETRAINED_MODEL_DIR:-$REPO_ROOT/models}"

# Pretraining checkpoints used by the public train config. Update URLs if the hosting path changes.
# Keep `dry_run` import-only so users can verify the environment without downloading large files.
if [[ "$MODE" == "train" || "$MODE" == "train_eval" ]]; then
  ensure_model "ConvNext-B-RGB.pth" "${CONVNEXT_RGB_URL:-https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/ConvNext-B-RGB.pth}"
  ensure_model "localization_checkpoint.pth" "${LOCALIZATION_CHECKPOINT_URL:-https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth}"
fi

read -r -a BACKBONES <<< "${BACKBONES:-tu-convnext_base}"
run_current_variant() {
  echo "[variant] name=${VARIANT_NAME:-custom} backbone=$CLOTH_COMPETITION_BACKBONE depth=$DEPTH_ENCODING crop=$CROP_MODE dataset=$CLOTH_COMPETITION_ICRA2024_DATASET"

  if [[ "$MODE" == "dry_run" ]]; then
    python - <<'PY'
import importlib, os
for name in ['config.cloth_6dof.train', 'config.cloth_6dof.test']:
    module = importlib.import_module(name)
    args = module.get_args()
    print(name, 'save_dir=', args.get('save_dir'), 'crop=', args.get('dataset', args.get('train_dataset', {})).get('kwargs', {}).get('crop_by_robot_positions'))
PY
  fi

  if [[ "$MODE" == "train" || "$MODE" == "train_eval" ]]; then
    python -m train_polar --configs skip_if_exists=True
  fi

  if [[ "$MODE" == "eval" || "$MODE" == "train_eval" ]]; then
    python -m test_polar --configs eval_epoch="${EVAL_EPOCH:-}" skip_if_exists=True
  fi
}

export DATASET=cloth_6dof
for backbone in "${BACKBONES[@]}"; do
  export CLOTH_COMPETITION_BACKBONE="$backbone"
  if [[ -n "${DEPTH_ENCODINGS:-}" || -n "${CROP_MODES:-}" ]]; then
    read -r -a DEPTH_LIST <<< "${DEPTH_ENCODINGS:-off 3d_arm_plane}"
    read -r -a CROP_LIST <<< "${CROP_MODES:-robot off}"
    for depth in "${DEPTH_LIST[@]}"; do
      export DEPTH_ENCODING="$depth"
      for crop in "${CROP_LIST[@]}"; do
        export CROP_MODE="$crop"
        VARIANT_NAME=custom run_current_variant
      done
    done
  else
    read -r -a VARIANT_LIST <<< "${VARIANTS:-crop_rgb nocrop_rgb crop_rgbd}"
    for variant in "${VARIANT_LIST[@]}"; do
      case "$variant" in
        crop_rgb)
          export DEPTH_ENCODING=off
          export CROP_MODE=robot
          ;;
        nocrop_rgb)
          export DEPTH_ENCODING=off
          export CROP_MODE=off
          ;;
        crop_rgbd)
          export DEPTH_ENCODING=3d_arm_plane
          export CROP_MODE=robot
          ;;
        *)
          echo "Unknown VARIANTS entry: $variant" >&2
          echo "Supported released variants: crop_rgb nocrop_rgb crop_rgbd" >&2
          exit 2
          ;;
      esac
      VARIANT_NAME="$variant" run_current_variant
    done
  fi
done
