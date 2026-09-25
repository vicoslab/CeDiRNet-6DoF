# Dataset setup

CeDiRNet-6DoF uses the ICRA 2024 cloth-competition dataset plus a small sidecar ZIP that adds the cloth segmentation masks used by the public train/eval code.

## 1. Download the original dataset

Download the original dataset from Zenodo:

```bash
wget -O ICRA_2024_cloth_competition_dataset.zip \
  'https://zenodo.org/records/14621179/files/ICRA_2024_cloth_competition_dataset.zip?download=1'
```

Unpack it so that your dataset root contains the competition dataset folders, for example:

```bash
unzip ICRA_2024_cloth_competition_dataset.zip -d /path/to/ClothCompetitionICRA2024
```

The public configs expect a dataset root containing folders such as:

```text
/path/to/ClothCompetitionICRA2024/datasets/
  cloth_competition_dataset_0000/sample_000000/...
  cloth_competition_dataset_0001/sample_000000/...
```

Set:

```bash
export CLOTH_COMPETITION_ICRA2024_DATASET=/path/to/ClothCompetitionICRA2024/datasets
```

## 2. Apply the CeDiRNet-6DoF sidecar masks

Download the sidecar ZIP:

```bash
wget -O icra2024_dataset_fixes.zip \
  https://data.vicos.si/skokec/rtfm/CeDiRNet-6DoF/icra2024_dataset_fixes.zip
```

Unpack it directly into the dataset root:

```bash
cd "$CLOTH_COMPETITION_ICRA2024_DATASET"
unzip /path/to/icra2024_dataset_fixes.zip
```

After unpacking, each public training/evaluation sample should have:

```text
sample_XXXXXX/observation_start/cloth_segmentation.png
```

The sidecar ZIP intentionally contains only these cloth segmentation masks. It does **not** contain `grasp/` subfolders, original RGB/depth images, point clouds, videos, camera calibration, robot poses, result files, or `cloth_segmentation_sam.png` masks.

## Training and validation split

The released configs select **440 training samples** (100 from dataset `0000`,
340 from `0001`) and **63 validation samples** (25 from `0000`, 38 from `0001`).
The `0001` validation items are references `08`, `14`, and `25`. The splits are
disjoint and together cover all 503 mask files.

Use the **complete original Zenodo archive**, not a partial competition-data
mirror. It supplies the RGB/depth images, calibration, robot poses, and
`grasp/grasp_pose.json` and `grasp/grasp_annotation.json` annotations required by
the loader. The mask archive is only a supplement, not a standalone dataset.
Segmentation masks are required by the released training and validation configs.

## Sidecar artifact details

The local generated copy is ignored by git:

```text
dataset/icra2024_dataset_fixes.zip
samples: 503
sidecar files: 503 (cloth_segmentation.png only; no grasp/ folders)
sha256: b6dd0a967d1df0749119d7d92e9e874d0968cce353afa12a675f1cc5ab3877e3
```
