# SRPL-SFDA on ABDOMINAL (BTCV ↔ CHAOST2)

Reproduction of the SRPL-SFDA method on the ABDOMINAL cross-modality dataset with
**bidirectional** adaptation support.

| Direction | Source | Target | Description |
|-----------|--------|--------|-------------|
| **BTCV→CHAOST2** | BTCV (CT) | CHAOST2 (MRI T2-SPIR) | CT→MR, primary direction |
| **CHAOST2→BTCV** | CHAOST2 (MRI) | BTCV (CT) | MR→CT, reverse direction |

## Dataset

| Property | BTCV | CHAOST2 |
|----------|:----:|:-------:|
| Modality | CT | MRI T2-SPIR |
| Volumes | 30 | 20 |
| Resolution | 512×512 | 256×256 / 320×320 |
| Classes | 5 (BG, Liver, R.Kidney, L.Kidney, Spleen) |
| Format | npz slices (`img` + `label`) |
| Train/Test | 24 / 6 | 16 / 4 |

Data location: `datasets/datasets/ABDOMINAL/processed_DDFP/`

## Environment

```bash
# Core dependencies
Python 3.10
PyTorch >= 1.8
segment-anything
h5py, GeodisTK, medpy, SimpleITK, nibabel, scipy, opencv-python, tqdm, tensorboardX

# Install
pip install segment-anything h5py GeodisTK medpy SimpleITK nibabel opencv-python tqdm tensorboardX
```

## Method Overview

SRPL-SFDA (SAM-guided Reliable Pseudo-Labels for Source-Free Domain Adaptation) is a
source-free domain adaptation method for medical image segmentation. The key innovation
is **test-time tri-branch intensity enhancement** — applying three intensity transformations
to target images and ensembling the source model's predictions to produce robust pseudo-labels.

The original paper targets **binary segmentation** (fetal brain, 1 foreground class).
We reproduce all applicable components for **5-class multi-organ segmentation**.
The SAM bbox refinement step is designed for binary segmentation (single bbox covering
the sole foreground object) and cannot be directly extended to multi-class without
introducing class-confusion errors. See [SAM and Multi-Class](#sam-and-multi-class) below.

### Pipeline

```
Phase 0: Source Model Training
    UNet supervised on source domain, 5-class Dice+CE loss
    Output: results/{SOURCE}2{TARGET}/source_model/best_model.pth
    ↓
Phase 1: Test-Time Tri-branch Intensity Enhancement + Ensemble PL
    For each target slice:
      1. Apply 3 intensity transformations (same as original paper):
         a) Histogram equalization (cv2.equalizeHist)
         b) Gamma correction γ=0.6 ("reduce Darkness")
         c) Gamma correction → source stats mean=0.5, std=0.29 ("reduce Style")
      2. Source model predicts class probabilities on each enhanced version
      3. Average the 3 probability maps → ensemble pseudo-label
      4. Compute uncertainty = normalized entropy of ensemble probabilities
    Output: data/ABDOMINAL/{SOURCE}2{TARGET}/enhanced_pl/
    ↓
Phase 2: RPL Selection + Fine-tune (Ablation, Step 3)
    Train on target with supervised loss on reliable regions only:
      reliable_mask = (uncertainty < T_fix × ln(C))
      L = CE(reliable) + Dice(reliable)
    Output: results/{SOURCE}2{TARGET}/step3_RPL/
    ↓
Phase 3: RPL + Entropy Minimization (Full Method, Step 4)
    Add entropy minimization on unreliable regions:
      L = L_sup(reliable) + λ × L_EM(unreliable)
    Output: results/{SOURCE}2{TARGET}/step4_EM/
    ↓
Evaluation: Dice, HD95, ASSD per class on target test set
```

## Quick Start

```bash
cd /opt/data/private/SRPL-SFDA
export PYTHONPATH=$PWD:$PYTHONPATH

# CT → MR (default)
bash run_abdominal.sh
# or equivalently:
bash run_btcv2chaost2.sh

# MR → CT (reverse)
bash run_chaost22btcv.sh
```

## Step-by-Step Execution

All scripts accept `--source_domain BTCV` or `--source_domain CHAOST2` to control
the adaptation direction. The direction determines which dataset is source vs target
and where outputs are saved.

### Phase 0: Source Model Training

```bash
# CT→MR: train on BTCV
python train_code/abdominal/0_source_train.py \
    --source_domain BTCV \
    --batch_size 32 \
    --max_epochs 100

# MR→CT: train on CHAOST2
python train_code/abdominal/0_source_train.py \
    --source_domain CHAOST2 \
    --batch_size 32 \
    --max_epochs 100
```

Trains a UNet (in_chns=1, class_num=5) on the source domain with Dice+CE loss.
Saves checkpoints to `results/{SOURCE}2{TARGET}/source_model/`.

### Phase 1: Tri-branch Intensity Enhancement + Ensemble

```bash
# CT→MR: enhance CHAOST2 (MRI) slices
python train_code/abdominal/1_1_intensity_enhancement.py \
    --source_domain BTCV \
    --source_model results/BTCV2CHAOST2/source_model/best_model.pth \
    --output_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl \
    --split train
python train_code/abdominal/1_1_intensity_enhancement.py \
    --source_domain BTCV \
    --source_model results/BTCV2CHAOST2/source_model/best_model.pth \
    --output_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl \
    --split val
```

This is the core innovation of SRPL-SFDA. For each target slice, three intensity
transformations are applied (identical to the original paper's `1_1`~`1_4` scripts):

| Branch | Transform | Paper Script | Description |
|--------|-----------|:---:|-------------|
| rD | Gamma γ=0.6 | `1_2_image_trans_rD.py` | Brightens dark regions, enhances soft tissue |
| rS | Gamma → μ=0.5, σ=0.29 | `1_3_image_trans_rS.py` | Matches intensity statistics to source domain via Nelder-Mead optimization |
| Equal | Histogram equalization | `1_1_image_trans_equal.py` | Maximizes local contrast uniformly |

The source model predicts class probabilities on all 3 enhanced versions,
which are then averaged (ensemble) to produce the final pseudo-label.
Uncertainty is the normalized entropy of the averaged probability distribution.

**Why ensemble helps**: Different enhancements highlight different anatomical features.
Averaging reduces the impact of any single branch's mistakes, producing more
robust pseudo-labels than any single prediction.

Output: per-slice `.npz` files with keys `pseudo_label` (uint8) and `uncertainty_map` (float32).

### Phase 2: RPL Selection (Step 3 — Ablation)

```bash
# CT→MR
python train_code/abdominal/3_train_RPL_selectRPL.py \
    --source_domain BTCV \
    --source_model results/BTCV2CHAOST2/source_model/best_model.pth \
    --exp BTCV2CHAOST2/step3_RPL \
    --max_iterations 3000 \
    --pseudo_label_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl/train
```

Reliable Pseudo-Label selection: pixels with ensemble uncertainty below
`T_fix × ln(C) = 0.5 × ln(5) ≈ 0.805` are treated as reliable.
Supervised loss (CE + weighted Dice) is applied only on reliable regions.

### Phase 3: RPL + Entropy Minimization (Step 4 — Full Method)

```bash
# CT→MR
python train_code/abdominal/4_train_RPL_add_EM.py \
    --source_domain BTCV \
    --source_model results/BTCV2CHAOST2/source_model/best_model.pth \
    --exp BTCV2CHAOST2/step4_EM \
    --max_iterations 3000 \
    --lameta_fix 0.1 \
    --pseudo_label_dir data/ABDOMINAL/BTCV2CHAOST2/enhanced_pl/train
```

Adds entropy minimization on unreliable regions:
`L = L_sup(reliable_regions) + λ × L_EM(unreliable_regions)`

### Evaluation

```bash
# Source-only (baseline, no adaptation)
python train_code/abdominal/test_metrics.py \
    --source_domain BTCV \
    --model_path results/BTCV2CHAOST2/source_model/best_model.pth \
    --domain target --split test

# Step 3 (RPL ablation)
python train_code/abdominal/test_metrics.py \
    --source_domain BTCV \
    --model_path results/BTCV2CHAOST2/step3_RPL/best_model.pth \
    --domain target --split test

# Step 4 (Full SRPL-SFDA)
python train_code/abdominal/test_metrics.py \
    --source_domain BTCV \
    --model_path results/BTCV2CHAOST2/step4_EM/best_model.pth \
    --domain target --split test
```

Reports per-class Dice, HD95, and ASSD.

## SAM and Multi-Class

The original SRPL-SFDA SAM bbox refinement (`2_1_everything_bbox.py`, `2_3_Med_SAM_bbox_seg.py`)
operates as follows:

1. Take the binary (single-class) ensemble pseudo-label
2. Extract **one** bbox covering the entire foreground via `get_Centroid_Endpoint_seed_points_2d()`
3. SAM refines the boundary → one binary mask
4. No class confusion possible (only 1 foreground class)

For multi-class (5 organs), this approach breaks down:

- Per-class bboxes of adjacent organs inevitably overlap
- SAM is class-agnostic — it segments the most prominent object in a bbox region
- When Liver's bbox overlaps Spleen, SAM may segment Spleen and label it "Liver"
- Sequential assignment causes later classes to overwrite earlier ones

We have verified experimentally that both SAM ViT-H and MedSAM ViT-B degrade
pseudo-label quality on this 5-class task (Dice drops 15-35% across all organs).

The SAM step is therefore excluded from this reproduction. The ensemble pseudo-labels
from Phase 1 are used directly for RPL training.

## Directory Structure

```
results/
├── BTCV2CHAOST2/
│   ├── source_model/    # Phase 0 output
│   ├── step3_RPL/       # Phase 2 output
│   └── step4_EM/        # Phase 3 output
└── CHAOST22BTCV/
    ├── source_model/
    ├── step3_RPL/
    └── step4_EM/

data/ABDOMINAL/
├── BTCV2CHAOST2/
│   └── enhanced_pl/     # Phase 1 output
│       ├── train/
│       └── val/
├── CHAOST22BTCV/
│   └── enhanced_pl/
│       ├── train/
│       └── val/
└── data_split/
```

## Key Hyperparameters

| Parameter | Value | Phase | Description |
|-----------|-------|:-----:|-------------|
| `T_fix` | 0.5 | Step 3 | Entropy threshold = T_fix × ln(C) |
| `threshold` | 0.6 | Step 4 | Reliability threshold |
| `lameta_fix` | 0.1 | Step 4 | EM loss weight |
| `base_lr` | 1e-3 | Phase 0/Step 4 | Learning rate |
| `base_lr` | 1e-4 | Step 3 | Learning rate (lower for fine-tune) |
| `batch_size` | 32-64 | All | Training batch size |
| `patch_size` | [256, 256] | All | Input resolution |
| `max_iterations` | 3000 | Step 3/4 | Training iterations |

## File Structure

```
SRPL-SFDA/
├── dataloaders/
│   └── abdominal_dataset.py            # ABDOMINAL npz dataloader (bidirectional)
├── train_code/
│   └── abdominal/
│       ├── 0_source_train.py           # Phase 0: source model training
│       ├── 1_1_intensity_enhancement.py # Phase 1: tri-branch enhancement + ensemble
│       ├── 3_train_RPL_selectRPL.py    # Phase 2: RPL selection + fine-tune
│       ├── 4_train_RPL_add_EM.py       # Phase 3: RPL + entropy minimization
│       └── test_metrics.py             # Evaluation (Dice, HD95, ASSD)
├── datasets/datasets/ABDOMINAL/
│   └── processed_DDFP/                 # npz slices + metadata.json
├── data/ABDOMINAL/
│   ├── BTCV2CHAOST2/enhanced_pl/       # Phase 1 output (CT→MR)
│   ├── CHAOST22BTCV/enhanced_pl/       # Phase 1 output (MR→CT)
│   └── data_split/                     # Train/val/test split info
├── results/                            # Training outputs (per-direction)
├── run_abdominal.sh                    # Default pipeline (BTCV→CHAOST2)
├── run_btcv2chaost2.sh                 # CT→MR pipeline
├── run_chaost22btcv.sh                 # MR→CT pipeline
└── README_ABDOMINAL.md                 # This file
```

## Reference

- Original SRPL-SFDA code: `train_code/fetal_brain/` (unchanged)
- SAM: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
