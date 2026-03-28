"""
Stage 2-2: Concatenate three intensity-transformed images as a 3-channel RGB PNG.
Each PNG serves as the visual input to MedSAM in stage 2-3.

The 3 channels are:
  R = histogram-equalized image
  G = gamma-dark (γ=0.6) image
  B = gamma-bright (γ=1.4) image

Reads from:  {data_dir}/{domain}/slices/             → img (original float)
Writes to:   {data_dir}/{domain}/sam_input_pngs/
             vol_XXXX_slice_YYYY.png
"""
import argparse
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--domain", type=str, default="CHAOST2")
    p.add_argument("--gamma_rD", type=float, default=0.6)
    p.add_argument("--gamma_rS", type=float, default=1.4)
    return p.parse_args()


def to_uint8(img_2d: np.ndarray) -> np.ndarray:
    vmin, vmax = img_2d.min(), img_2d.max()
    return ((img_2d - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)


def hist_equalize(img_2d: np.ndarray) -> np.ndarray:
    uint8 = to_uint8(img_2d)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(uint8)


def gamma_transform(img_2d: np.ndarray, gamma: float) -> np.ndarray:
    vmin, vmax = img_2d.min(), img_2d.max()
    norm = (img_2d - vmin) / (vmax - vmin + 1e-8)
    corrected = np.power(norm + 1e-10, gamma)
    return (corrected * 255).astype(np.uint8)


def main():
    args = parse_args()
    src_dir = os.path.join(args.data_dir, args.domain, "slices")
    out_dir = os.path.join(args.data_dir, args.domain, "sam_input_pngs")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npz")])
    print(f"Creating {len(files)} RGB PNGs for SAM input → {out_dir}")

    for fname in files:
        img = np.load(os.path.join(src_dir, fname))["img"].astype(np.float32)

        ch_eq = hist_equalize(img)
        ch_rD = gamma_transform(img, args.gamma_rD)
        ch_rS = gamma_transform(img, args.gamma_rS)

        # Stack as BGR for OpenCV (R=eq, G=rD, B=rS)
        rgb = np.stack([ch_eq, ch_rD, ch_rS], axis=-1)  # (H, W, 3)

        out_name = fname.replace(".npz", ".png")
        cv2.imwrite(os.path.join(out_dir, out_name), rgb)

    print("Stage 2-2 (concat RGB PNG) done.")


if __name__ == "__main__":
    main()
