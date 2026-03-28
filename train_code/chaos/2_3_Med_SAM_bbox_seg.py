"""
Stage 2-3: MedSAM per-class BBox segmentation for multi-class CHAOS.

For each slice in slices_avg_pl/:
  For each foreground class c in {1=Liver, 2=R.Kidney, 3=L.Kidney, 4=Spleen}:
    1. Check if class c is present in the averaged pseudo-label.
    2. Compute tight bounding box around class c pixels.
    3. Feed the RGB PNG (from stage 2-2) + bbox to MedSAM.
    4. Use the SAM binary mask as the refined label for class c.
  Merge all 4 class masks into a single multi-class label.
  Conflict resolution: later classes (higher index) overwrite earlier ones.

Reads:
  {data_dir}/{domain}/slices_avg_pl/    → img, label (pseudo-PL), gt, uncertainty_map
  {data_dir}/{domain}/sam_input_pngs/   → RGB PNGs (from 2_2)

Writes:
  {data_dir}/{domain}/slices_sam_pl/    → img, label (SAM-refined), gt, uncertainty_map
"""
import argparse
import os
import sys

import numpy as np
import torch
from skimage.io import imread

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# SAM / MedSAM import (segment_anything must be installed)
try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as e:
    raise ImportError(
        "segment_anything is not installed. "
        "See https://github.com/facebookresearch/segment-anything"
    ) from e

CHAOS_CLASSES = {1: "Liver", 2: "R.Kidney", 3: "L.Kidney", 4: "Spleen"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--domain", type=str, default="CHAOST2")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to MedSAM / SAM checkpoint (.pth)")
    p.add_argument("--model_type", type=str, default="vit_b")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--min_class_pixels", type=int, default=10,
                   help="Minimum pixels for a class to be eligible for SAM refinement")
    p.add_argument("--bbox_margin", type=int, default=5,
                   help="Margin (pixels) to expand bbox before SAM query")
    return p.parse_args()


def get_bbox(binary_mask: np.ndarray, margin: int, h: int, w: int):
    """Return (x0, y0, x1, y1) bounding box with margin, clipped to image bounds."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    x0 = max(0, cmin - margin)
    y0 = max(0, rmin - margin)
    x1 = min(w - 1, cmax + margin)
    y1 = min(h - 1, rmax + margin)
    return np.array([x0, y0, x1, y1])


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build SAM predictor
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(device)
    predictor = SamPredictor(sam)

    base = os.path.join(args.data_dir, args.domain)
    pl_dir = os.path.join(base, "slices_avg_pl")
    png_dir = os.path.join(base, "sam_input_pngs")
    out_dir = os.path.join(base, "slices_sam_pl")
    os.makedirs(out_dir, exist_ok=True)

    assert os.path.exists(pl_dir), f"Missing: {pl_dir}  (run stage 1_4 first)"
    assert os.path.exists(png_dir), f"Missing: {png_dir}  (run stage 2_2 first)"

    files = sorted([f for f in os.listdir(pl_dir) if f.endswith(".npz")])
    print(f"Refining {len(files)} slices with MedSAM → {out_dir}")

    for fname in files:
        data = np.load(os.path.join(pl_dir, fname))
        img_2d = data["img"].astype(np.float32)
        pseudo_label = data["label"].astype(np.uint8)
        gt = data["gt"].astype(np.uint8)
        uncertainty_map = data["uncertainty_map"].astype(np.float32)
        h, w = img_2d.shape

        # Load the 3-channel RGB PNG for SAM
        png_path = os.path.join(png_dir, fname.replace(".npz", ".png"))
        if not os.path.exists(png_path):
            # No PNG → copy averaged PL without SAM refinement
            np.savez_compressed(
                os.path.join(out_dir, fname),
                img=img_2d, label=pseudo_label, gt=gt, uncertainty_map=uncertainty_map,
            )
            continue

        rgb = imread(png_path)[:, :, :3]   # (H, W, 3) uint8
        predictor.set_image(rgb)

        refined_label = np.zeros_like(pseudo_label)  # background = 0

        for cls in sorted(CHAOS_CLASSES.keys()):
            binary = (pseudo_label == cls)
            if binary.sum() < args.min_class_pixels:
                continue  # class absent or too small → skip

            bbox = get_bbox(binary, args.bbox_margin, h, w)
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox[None, :],   # shape (1, 4)
                    multimask_output=False,
                )
                sam_mask = masks[0]  # (H, W) bool
            except Exception as exc:
                print(f"  SAM error on {fname} cls={cls}: {exc}")
                sam_mask = binary  # fallback to averaged PL

            # Assign class label (later classes overwrite conflicts)
            refined_label[sam_mask] = cls

        np.savez_compressed(
            os.path.join(out_dir, fname),
            img=img_2d,
            label=refined_label.astype(np.uint8),
            gt=gt,
            uncertainty_map=uncertainty_map,
        )

    print("Stage 2-3 (MedSAM multi-class refinement) done.")


if __name__ == "__main__":
    main()
