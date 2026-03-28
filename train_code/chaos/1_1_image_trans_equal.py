"""
Stage 1-1: Apply histogram equalization per slice and run source model.
Reads from:  {data_dir}/{domain}/slices/vol_XXXX_slice_YYYY.npz  (img, label=GT)
Writes to:   {data_dir}/{domain}/slices_pred_equal/vol_XXXX_slice_YYYY.npz  (pred_prob [C,H,W])
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from networks.net_factory import net_factory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to processed_DDFP directory (contains metadata.json)")
    p.add_argument("--domain", type=str, default="CHAOST2")
    p.add_argument("--source_model", type=str, required=True,
                   help="Path to source-domain pretrained model (.pth)")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()


def hist_equalize(img_2d: np.ndarray) -> np.ndarray:
    """Min-max → uint8 → CLAHE → float32 [0,1]."""
    vmin, vmax = img_2d.min(), img_2d.max()
    uint8 = ((img_2d - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(uint8)
    return eq.astype(np.float32) / 255.0


@torch.no_grad()
def run_model(net, img_2d: np.ndarray, patch_size: int) -> np.ndarray:
    """Return softmax prediction array of shape (C, H, W)."""
    h, w = img_2d.shape
    # Resize to patch_size
    from scipy.ndimage.interpolation import zoom
    scale_h, scale_w = patch_size / h, patch_size / w
    inp = zoom(img_2d, (scale_h, scale_w), order=1)
    tensor = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    net.eval()
    prob = torch.softmax(net(tensor), dim=1).squeeze(0)  # (C, patch, patch)
    # Resize back to original size
    prob_np = prob.cpu().numpy()
    if h != patch_size or w != patch_size:
        from scipy.ndimage.interpolation import zoom as ndzoom
        prob_np = np.stack([
            ndzoom(prob_np[c], (h / patch_size, w / patch_size), order=1)
            for c in range(prob_np.shape[0])
        ], axis=0)
    return prob_np  # (C, H, W)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load metadata
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    # Build model
    net = net_factory("unet2d", in_chns=1, class_num=args.num_classes)
    net.load_state_dict(torch.load(args.source_model, map_location="cuda:0"))
    net.eval()

    src_dir = os.path.join(args.data_dir, args.domain, "slices")
    out_dir = os.path.join(args.data_dir, args.domain, "slices_pred_equal")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npz")])
    print(f"Processing {len(files)} slices → {out_dir}")

    for fname in files:
        data = np.load(os.path.join(src_dir, fname))
        img = data["img"].astype(np.float32)
        img_eq = hist_equalize(img)
        pred_prob = run_model(net, img_eq, args.patch_size)
        np.savez_compressed(os.path.join(out_dir, fname), pred_prob=pred_prob)

    print("Stage 1-1 (equal) done.")


if __name__ == "__main__":
    main()
