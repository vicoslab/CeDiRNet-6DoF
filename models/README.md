# Released checkpoint provenance

All three downloadable files have been checked by SHA256 against their original
training checkpoints. The RGB files also exactly match the robot evaluation
checkpoint store. Hashes, sizes, download URLs, and inference configurations are
in [`model_definitions.json`](model_definitions.json).

| Public variant | Training/evaluation model key | Network channels | Arm-plane depth mean/std |
| --- | --- | ---: | --- |
| `crop_rgb` | `v2+seg+rand_bg+cropping` | 3 | not a network input |
| `nocrop_rgb` | `v2+seg+rand_bg` | 3 | not a network input |
| `crop_rgbd` | `v2+seg+rand_bg+cropping+3D-arm-plane-depth` | 4 | **0 / 1** |

All use ConvNeXt-base, segmentation supervision and background randomization.
Use **`crop_rgb` as the recommended starting point**: it corresponds to the
best RGB model described in the paper. `nocrop_rgb` is provided for setups that
cannot supply a valid robot-workspace crop. The RGB-D model is an additional
variant; it is **not** the separate `+std=0.5` RGB-D ablation evaluated in RUNS B.
Do not assign that ablation's numbers to this download.

The public RGB-D file is the completed 100-epoch model (stored epoch index 99).
Its saved training parameters specify `depth_mean=0`, `depth_std=1` and four
input channels. Normalization is applied in preprocessing, not recovered from
the checkpoint state dictionary. Using `depth_std=0.5` with this file changes the
network input and is incorrect. The release train/eval/inference defaults use 1.

The supplied inference configs describe model construction and preprocessing;
the entrypoint selects the fixed released architecture and tests verify parity
with these configs. Editing the manifest's `config` field alone does not change
the entrypoint. Custom architectures require corresponding code changes.

## Reproduction boundaries

- Reported coverage and grasp success are physical robot measurements, not
  offline image metrics. The README identifies their experiment and conditions.
- The released train helper initializes from the public CeDiRNet-3DoF checkpoint.
  The paper used an extended pretraining dataset/protocol. Exact reproduction
  of that pretraining stage is not established by this release.
- COCO background replacement requires both documented environment variables.
  The list must contain one image filename **including its extension** per line,
  without blank/comment lines.
- The recovered training configuration uses a **1.8 m** depth threshold for
  background replacement. The paper prose says **1.2 m**; this release preserves
  the recovered configuration rather than silently changing it.
- Different random seeds, library versions, and robot conditions can change
  results; no exact retraining or physical-robot reproduction is claimed.

Only load trusted checkpoints. PyTorch model files may contain pickle data;
checksums establish integrity against this manifest, not safety of arbitrary
third-party files.
