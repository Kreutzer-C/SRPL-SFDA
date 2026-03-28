# -*- coding: utf-8 -*-
"""
CHAOS dataset loader for SRPL-SFDA pipeline.
Reads from MemProp-style processed_DDFP NPZ slices.

Expected directory layout (under base_dir):
  {domain}/slices/                    # Original MemProp slices: img (float H,W), label (int H,W, GT)
  {domain}/slices_sam_pl/             # After full preprocessing:
      vol_{XXXX}_slice_{YYYY}.npz     #   img, label (pseudo), gt, uncertainty_map

Label mapping (CHAOS, 5 classes):
  0=Background, 1=Liver, 2=R.Kidney, 3=L.Kidney, 4=Spleen
"""
from __future__ import print_function, division

import json
import os

import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Training dataset: individual 2D slices with pseudo-labels + uncertainty
# ---------------------------------------------------------------------------
class ChaosSliceDataset(Dataset):
    """
    Load 2D slices from `slices_sam_pl/` (SAM-refined pseudo-labels).
    Falls back to `slices_avg_pl/` when SAM results are unavailable.
    Each NPZ contains: img, label (pseudo), gt, uncertainty_map.
    """

    def __init__(self, base_dir, domain, metadata, split="train",
                 pl_subdir="slices_sam_pl", transform=None):
        self.base_dir = base_dir
        self.domain = domain
        self.split = split
        self.transform = transform

        # Determine slice directory
        self.slice_dir = os.path.join(base_dir, domain, pl_subdir)
        if not os.path.exists(self.slice_dir):
            # Try fallback to averaged PL before SAM
            fallback = os.path.join(base_dir, domain, "slices_avg_pl")
            assert os.path.exists(fallback), (
                f"Slice directory not found: {self.slice_dir}\n"
                f"Fallback also missing: {fallback}\n"
                "Run preprocessing stages 1_1→1_4 (and 2_2→2_3) first."
            )
            self.slice_dir = fallback

        # Collect files for the requested split
        case_ids = set(metadata["splits"][domain][split])
        self.files = sorted([
            os.path.join(self.slice_dir, f)
            for f in os.listdir(self.slice_dir)
            if f.endswith(".npz") and f.split("_")[1] in case_ids
        ])
        assert len(self.files) > 0, (
            f"No NPZ files found in {self.slice_dir} for split '{split}'"
        )
        print(f"[ChaosSliceDataset] domain={domain} split={split} "
              f"slices={len(self.files)} from {self.slice_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        fname = os.path.basename(fpath)
        case_name = fname.split("_")[1]            # "XXXX"
        slice_name = fname.split("_")[3].split(".")[0]  # "YYYY"

        data = np.load(fpath)
        image = data["img"].astype(np.float32)
        label = data["label"].astype(np.uint8)          # pseudo-label
        gt = data["gt"].astype(np.uint8)                 # ground truth
        uncertainty_map = data["uncertainty_map"].astype(np.float32)

        sample = {
            "image": image,
            "label": label,
            "gt": gt,
            "uncertainty_map": uncertainty_map,
            "image_name": fname.replace(".npz", ""),
            "idx": idx,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Validation dataset: 3D volumes reconstructed from original GT slices
# ---------------------------------------------------------------------------
class ChaosVolDataset(Dataset):
    """
    Load complete 3D volumes from `slices/` for validation / testing.
    Groups slices by case_id, stacks them in slice order.
    Returns image (D,H,W) and gt (D,H,W) using the original GT labels.
    """

    def __init__(self, base_dir, domain, metadata, split="val", transform=None):
        self.base_dir = base_dir
        self.domain = domain
        self.split = split
        self.transform = transform

        slice_dir = os.path.join(base_dir, domain, "slices")
        assert os.path.exists(slice_dir), (
            f"Original slice directory not found: {slice_dir}"
        )

        case_ids = metadata["splits"][domain][split]
        # Build: case_id -> sorted list of (slice_idx, fpath)
        case_map = {cid: [] for cid in case_ids}
        for f in os.listdir(slice_dir):
            if not f.endswith(".npz"):
                continue
            parts = f.split("_")
            cid = parts[1]
            sidx = int(parts[3].split(".")[0])
            if cid in case_map:
                case_map[cid].append((sidx, os.path.join(slice_dir, f)))

        self.volumes = []
        for cid in sorted(case_ids):
            slices = sorted(case_map[cid], key=lambda x: x[0])
            assert len(slices) > 0, f"No slices found for case {cid}"
            self.volumes.append((cid, [fp for _, fp in slices]))

        print(f"[ChaosVolDataset] domain={domain} split={split} "
              f"volumes={len(self.volumes)}")

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        case_id, fpaths = self.volumes[idx]
        imgs, gts = [], []
        for fpath in fpaths:
            data = np.load(fpath)
            imgs.append(data["img"].astype(np.float32))
            gts.append(data["label"].astype(np.uint8))  # GT in MemProp format

        image_3d = np.stack(imgs, axis=0)   # (D, H, W)
        gt_3d = np.stack(gts, axis=0)       # (D, H, W)

        # Min-max normalise the whole volume
        vmin, vmax = image_3d.min(), image_3d.max()
        image_3d = (image_3d - vmin) / (vmax - vmin + 1e-8)

        image_t = torch.from_numpy(image_3d).unsqueeze(0)  # (1, D, H, W)
        gt_t = torch.from_numpy(gt_3d.astype(np.int64))    # (D, H, W)

        return {
            "image": image_t,
            "gt": gt_t,
            "case_id": case_id,
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class TrainToTensor:
    """Resize 2D slice to output_size and convert to tensors."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        gt = sample["gt"]
        uncertainty_map = sample["uncertainty_map"]

        h, w = image.shape
        zh, zw = self.output_size[0] / h, self.output_size[1] / w
        image = zoom(image, (zh, zw), order=1)
        label = zoom(label, (zh, zw), order=0)
        gt = zoom(gt, (zh, zw), order=0)
        uncertainty_map = zoom(uncertainty_map, (zh, zw), order=1)

        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.uint8)),
            "gt": torch.from_numpy(gt.astype(np.uint8)),
            "uncertainty_map": torch.from_numpy(uncertainty_map.astype(np.float32)),
            "image_name": sample["image_name"],
            "idx": sample["idx"],
        }


class ValToTensor:
    """Convert val sample fields to tensors (no resize; handled inside test_single_volume)."""

    def __call__(self, sample):
        return {
            "image": torch.from_numpy(sample["image"].astype(np.float32)),
            "label": torch.from_numpy(sample["label"].astype(np.uint8)),
            "gt": torch.from_numpy(sample["gt"].astype(np.uint8)),
            "uncertainty_map": torch.from_numpy(
                sample["uncertainty_map"].astype(np.float32)
            ),
            "image_name": sample["image_name"],
            "idx": sample["idx"],
        }


# ---------------------------------------------------------------------------
# Helper: load metadata.json
# ---------------------------------------------------------------------------
def load_metadata(data_dir):
    meta_path = os.path.join(data_dir, "metadata.json")
    assert os.path.exists(meta_path), f"metadata.json not found: {meta_path}"
    with open(meta_path) as f:
        return json.load(f)
