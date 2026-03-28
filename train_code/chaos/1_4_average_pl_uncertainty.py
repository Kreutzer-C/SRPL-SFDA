"""
Stage 1-4: Average three softmax prediction sets → pseudo-label + uncertainty map.

Reads:
  {data_dir}/{domain}/slices/              → img (original image) + label (GT)
  {data_dir}/{domain}/slices_pred_equal/   → pred_prob [C,H,W]
  {data_dir}/{domain}/slices_pred_rD/      → pred_prob [C,H,W]
  {data_dir}/{domain}/slices_pred_rS/      → pred_prob [C,H,W]

Writes to {data_dir}/{domain}/slices_avg_pl/:
  vol_XXXX_slice_YYYY.npz  →  img, label (pseudo-label), gt, uncertainty_map
"""
import argparse
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--domain", type=str, default="CHAOST2")
    p.add_argument("--num_classes", type=int, default=5)
    return p.parse_args()


def normalized_entropy(prob: np.ndarray, C: int) -> np.ndarray:
    """Compute pixel-wise normalised entropy from prob array (C,H,W) → (H,W) in [0,1]."""
    ent = -np.sum(prob * np.log(prob + 1e-6), axis=0)  # (H, W)
    return ent / np.log(C)


def main():
    args = parse_args()
    C = args.num_classes

    base = os.path.join(args.data_dir, args.domain)
    src_dir = os.path.join(base, "slices")
    dir_eq = os.path.join(base, "slices_pred_equal")
    dir_rD = os.path.join(base, "slices_pred_rD")
    dir_rS = os.path.join(base, "slices_pred_rS")
    out_dir = os.path.join(base, "slices_avg_pl")
    os.makedirs(out_dir, exist_ok=True)

    for d in (dir_eq, dir_rD, dir_rS):
        assert os.path.exists(d), (
            f"Missing prediction directory: {d}\n"
            "Run 1_1, 1_2, 1_3 first."
        )

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npz")])
    print(f"Averaging {len(files)} slices → {out_dir}")
    missing = 0

    for fname in files:
        paths = [os.path.join(d, fname) for d in (dir_eq, dir_rD, dir_rS)]
        if not all(os.path.exists(p) for p in paths):
            missing += 1
            continue

        # Load original image and GT label
        orig = np.load(os.path.join(src_dir, fname))
        img = orig["img"].astype(np.float32)
        gt = orig["label"].astype(np.uint8)

        # Average the three softmax probability maps
        probs = [np.load(p)["pred_prob"].astype(np.float32) for p in paths]
        avg_prob = np.mean(probs, axis=0)  # (C, H, W)

        # Pseudo-label: argmax of averaged probabilities
        pseudo_label = np.argmax(avg_prob, axis=0).astype(np.uint8)  # (H, W)

        # Uncertainty: normalised entropy of averaged distribution
        uncertainty_map = normalized_entropy(avg_prob, C)  # (H, W), float32 [0,1]

        np.savez_compressed(
            os.path.join(out_dir, fname),
            img=img,
            label=pseudo_label,
            gt=gt,
            uncertainty_map=uncertainty_map.astype(np.float32),
        )

    if missing:
        print(f"  WARNING: {missing} slices skipped (missing predictions).")
    print("Stage 1-4 (average + uncertainty) done.")


if __name__ == "__main__":
    main()
