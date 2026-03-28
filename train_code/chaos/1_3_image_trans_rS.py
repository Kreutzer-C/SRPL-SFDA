"""
Stage 1-3: Apply random-bright gamma correction (γ=1.4) and run source model.
Reads from:  {data_dir}/{domain}/slices/
Writes to:   {data_dir}/{domain}/slices_pred_rS/   (pred_prob [C,H,W])
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from utils.model_utils import build_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--domain", type=str, default="CHAOST2")
    p.add_argument("--source_model", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=1.4,
                   help="Gamma > 1 darkens the image (bright-suppressed)")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--memprop_dir", type=str, default="/workspace/MemProp-SFDA")
    return p.parse_args()


def gamma_correction(img_2d: np.ndarray, gamma: float) -> np.ndarray:
    vmin, vmax = img_2d.min(), img_2d.max()
    norm = (img_2d - vmin) / (vmax - vmin + 1e-8)
    return np.power(norm + 1e-10, gamma).astype(np.float32)


@torch.no_grad()
def run_model(net, img_2d: np.ndarray, patch_size: int) -> np.ndarray:
    from scipy.ndimage.interpolation import zoom
    h, w = img_2d.shape
    inp = zoom(img_2d, (patch_size / h, patch_size / w), order=1)
    tensor = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    net.eval()
    prob = torch.softmax(net(tensor), dim=1).squeeze(0).cpu().numpy()
    if h != patch_size or w != patch_size:
        prob = np.stack([
            zoom(prob[c], (h / patch_size, w / patch_size), order=1)
            for c in range(prob.shape[0])
        ], axis=0)
    return prob


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        json.load(f)

    net = build_model_from_checkpoint(
        args.source_model, args.num_classes, memprop_dir=args.memprop_dir
    )

    src_dir = os.path.join(args.data_dir, args.domain, "slices")
    out_dir = os.path.join(args.data_dir, args.domain, "slices_pred_rS")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npz")])
    print(f"Processing {len(files)} slices (gamma={args.gamma}) → {out_dir}")

    for fname in files:
        img = np.load(os.path.join(src_dir, fname))["img"].astype(np.float32)
        img_gamma = gamma_correction(img, args.gamma)
        pred_prob = run_model(net, img_gamma, args.patch_size)
        np.savez_compressed(os.path.join(out_dir, fname), pred_prob=pred_prob)

    print("Stage 1-3 (rS) done.")


if __name__ == "__main__":
    main()
