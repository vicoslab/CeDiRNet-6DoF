#!/usr/bin/env bash
# Simple one-observation CeDiRNet-6DoF inference demo.
#
# Usage:
#   MODEL_CHECKPOINT=models/cedirnet6dof_crop_rgb.pth \
#   OBSERVATION_FOLDER=/path/to/sample_000000 \
#   ./scripts/run_demo_inference.sh
#
# Optional:
#   VARIANT=crop_rgb|nocrop_rgb|crop_rgbd
#   OUTPUT_DIR=/path/to/output
#   DEVICE=cuda|cpu
#   MODEL_DEFINITIONS=models/model_definitions.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-crop_rgb}"
DEVICE="${DEVICE:-cuda}"
OBSERVATION_FOLDER="${OBSERVATION_FOLDER:-${1:-}}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${2:-}}"
MODEL_DEFINITIONS="${MODEL_DEFINITIONS:-models/model_definitions.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "$OBSERVATION_FOLDER" ]]; then
  cat >&2 <<'EOF'
ERROR: OBSERVATION_FOLDER is required.

Example:
  MODEL_CHECKPOINT=models/cedirnet6dof_crop_rgb.pth \
  OBSERVATION_FOLDER=/path/to/ClothCompetitionICRA2024/datasets/cloth_competition_dataset_0000/sample_000000 \
  ./scripts/run_demo_inference.sh

You can also pass observation folder and checkpoint as positional arguments:
  ./scripts/run_demo_inference.sh /path/to/sample_000000 models/cedirnet6dof_crop_rgb.pth
EOF
  exit 2
fi

if [[ -z "$MODEL_CHECKPOINT" && -z "$MODEL_DEFINITIONS" ]]; then
  cat >&2 <<'EOF'
ERROR: Provide MODEL_CHECKPOINT or MODEL_DEFINITIONS.

Examples:
  MODEL_CHECKPOINT=models/cedirnet6dof_crop_rgb.pth ./scripts/run_demo_inference.sh /path/to/sample_000000
  MODEL_DEFINITIONS=models/model_definitions.json ./scripts/run_demo_inference.sh /path/to/sample_000000
EOF
  exit 2
fi

cmd=(python tools/infer_observation.py
  --observation-folder "$OBSERVATION_FOLDER"
  --variant "$VARIANT"
  --device "$DEVICE")

if [[ -n "$OUTPUT_DIR" ]]; then
  cmd+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$MODEL_CHECKPOINT" ]]; then
  cmd+=(--model-checkpoint "$MODEL_CHECKPOINT")
fi
if [[ -n "$MODEL_DEFINITIONS" ]]; then
  cmd+=(--model-definitions "$MODEL_DEFINITIONS")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
