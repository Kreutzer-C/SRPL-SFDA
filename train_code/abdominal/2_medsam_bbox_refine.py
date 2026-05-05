# -*- coding: utf-8 -*-
"""
Phase 2: MedSAM Bbox Refinement for Multi-Class Pseudo-Labels.
Adapted from the original SRPL-SFDA 2_1~2_3 pipeline for ABDOMINAL multi-class.

Original flow (binary):
  1. 3-channel RGB image: R=gamma-matched, G=gamma=0.6, B=histogram EQ
  2. Single bbox from binary mask via get_Centroid_Endpoint_seed_points_2d()
  3. SAM/MedSAM refines → one binary mask

Multi-class adaptation:
  - For each of the 4 organ classes independently:
    1. Get binary mask from ensemble pseudo-label
    2. Extract bbox from class mask
    3. MedSAM refines with bbox prompt → binary mask
  - Merge: where multiple classes claim a pixel, use ensemble probability to decide
"""
import os, sys, argparse, json
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.optimize import minimize

ROOT = '/opt/data/private/SRPL-SFDA'
sys.path.insert(0, ROOT)

from segment_anything import sam_model_registry, SamPredictor
from dataloaders.abdominal_dataset import NUM_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, default='BTCV', choices=['BTCV', 'CHAOST2'])
parser.add_argument('--pl_dir', type=str, default=None,
                    help='Path to ensemble pseudo-labels (Phase 1 output). Default: auto')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output dir for MedSAM-refined pseudo-labels. Default: auto')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--sam_checkpoint', type=str,
                    default=os.path.join(ROOT, 'checkpoints', 'medsam_vit_b.pth'))
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

SOURCE_KEY = args.source_domain
TARGET_KEY = 'CHAOST2' if SOURCE_KEY == 'BTCV' else 'BTCV'
DIRECTION = f'{SOURCE_KEY}2{TARGET_KEY}'

if args.pl_dir is None:
    args.pl_dir = os.path.join(ROOT, 'data', 'ABDOMINAL', DIRECTION, 'enhanced_pl', args.split)
if args.output_dir is None:
    args.output_dir = os.path.join(ROOT, 'data', 'ABDOMINAL', DIRECTION, 'medsam_pl', args.split)

os.makedirs(args.output_dir, exist_ok=True)
DEVICE = f'cuda:{args.gpu}'

# ---------------------------------------------------------------------------
# Intensity transforms (same as Phase 1)
# ---------------------------------------------------------------------------
def gamma_correction(image, gamma):
    return np.power(image + 1e-10, gamma)

def _optimize_gamma(processed_data, target_mean, target_std):
    def objective(gamma, image, tgt_mean, tgt_std):
        corrected = gamma_correction(image, gamma)
        return abs((corrected.mean() - tgt_mean) + (corrected.std() - tgt_std))
    result = minimize(objective, 1.0, args=(processed_data, target_mean, target_std),
                      method='Nelder-Mead')
    return result.x[0]

def apply_histogram_equalization(img_01):
    img_uint8 = (img_01 * 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_uint8)
    return equalized.astype(np.float32) / 255.0

def apply_gamma_fixed(img_01, gamma=0.6):
    mask = img_01 > 0
    if mask.sum() == 0:
        return img_01.copy()
    foreground = img_01[mask]
    fg_norm = (foreground - foreground.min()) / (foreground.max() - foreground.min() + 1e-10)
    corrected = gamma_correction(fg_norm, gamma)
    result = np.zeros_like(img_01)
    result[mask] = corrected
    return result

def apply_gamma_matched(img_01, target_mean=0.5, target_std=0.29):
    mask = img_01 > 0
    if mask.sum() == 0:
        return img_01.copy()
    foreground = img_01[mask]
    fg_norm = (foreground - foreground.min()) / (foreground.max() - foreground.min() + 1e-10)
    optimal_gamma = _optimize_gamma(fg_norm, target_mean, target_std)
    corrected = gamma_correction(fg_norm, optimal_gamma)
    result = np.zeros_like(img_01)
    result[mask] = corrected
    return result

# ---------------------------------------------------------------------------
# Data collection (same as Phase 1)
# ---------------------------------------------------------------------------
def _collect_target_slices(split_name, target_key):
    meta_path = os.path.join(ROOT, 'datasets', 'datasets', 'ABDOMINAL', 'processed_DDFP', 'metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    splits = meta['splits'][target_key]
    case_ids = splits['test'] if split_name in ('val', 'test') else splits['train']
    slice_dir = os.path.join(ROOT, 'datasets', 'datasets', 'ABDOMINAL', 'processed_DDFP', target_key, 'slices')
    samples = []
    for case_id in case_ids:
        prefix = f'vol_{case_id}_'
        for fname in sorted(os.listdir(slice_dir)):
            if fname.startswith(prefix) and fname.endswith('.npz'):
                samples.append((os.path.join(slice_dir, fname), fname.replace('.npz', '')))
    return samples

def load_and_preprocess(npz_path, patch_size):
    data = np.load(npz_path)
    img = data['img'].astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    if img.shape != tuple(patch_size):
        img = zoom(img, (patch_size[0] / img.shape[0], patch_size[1] / img.shape[1]), order=1)
    return img

# ---------------------------------------------------------------------------
# Bbox extraction (adapted for per-class multi-class)
# ---------------------------------------------------------------------------
def get_class_bbox(binary_mask, pad=5):
    """Extract bbox from a single-class binary mask. Returns (x0, y0, x1, y1) or None."""
    ys, xs = np.where(binary_mask > 0)
    if len(ys) == 0:
        return None
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad, binary_mask.shape[0] - 1)
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad, binary_mask.shape[1] - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1])

# ---------------------------------------------------------------------------
# Main MedSAM bbox refinement
# ---------------------------------------------------------------------------
def run():
    # Load MedSAM
    sam = sam_model_registry['vit_b'](checkpoint=args.sam_checkpoint)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    samples = _collect_target_slices(args.split, TARGET_KEY)
    print(f'Phase 2 MedSAM [{DIRECTION} | {args.split}]: {len(samples)} slices')
    print(f'  MedSAM: {args.sam_checkpoint}')
    print(f'  PL input: {args.pl_dir}')
    print(f'  Output: {args.output_dir}')

    for npz_path, fname in tqdm(samples, desc=f'MedSAM {TARGET_KEY} {args.split}'):
        # Load and enhance the target image
        img_01 = load_and_preprocess(npz_path, tuple(args.patch_size))

        # 3 intensity transforms
        img_eq = apply_histogram_equalization(img_01)       # B
        img_rd = apply_gamma_fixed(img_01, gamma=0.6)       # G
        img_rs = apply_gamma_matched(img_01)                 # R

        # Create 3-channel RGB: R=rS, G=rD, B=equal (matching original 2_2)
        rgb = np.stack([img_rs, img_rd, img_eq], axis=-1)   # [H, W, 3]
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Load ensemble pseudo-label
        pl_path = os.path.join(args.pl_dir, f'{fname}.npz')
        if not os.path.exists(pl_path):
            print(f'  WARNING: PL missing for {fname}, skipping')
            continue
        pl_data = np.load(pl_path)
        ensemble_pl = pl_data['pseudo_label'].astype(np.int64)  # [H, W], class indices

        # Set the RGB image for MedSAM
        predictor.set_image(rgb_uint8)

        # Per-class MedSAM bbox refinement
        refined_masks = {}  # class_idx -> binary mask
        for c in range(1, NUM_CLASSES):  # skip background (0)
            class_mask = (ensemble_pl == c).astype(np.uint8)
            if class_mask.sum() == 0:
                continue

            bbox = get_class_bbox(class_mask, pad=5)
            if bbox is None:
                continue

            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,
            )
            refined_masks[c] = masks[0].astype(np.float32)

        # Merge: start from ensemble PL, overlay MedSAM-refined class masks
        refined_pl = ensemble_pl.copy()
        for c, mask in refined_masks.items():
            refined_pl[mask > 0.5] = c

        # Save (carry forward the ensemble uncertainty map)
        uncertainty_map = pl_data.get('uncertainty_map', np.zeros_like(img_01, dtype=np.float32))
        np.savez_compressed(
            os.path.join(args.output_dir, f'{fname}.npz'),
            pseudo_label=refined_pl.astype(np.uint8),
            uncertainty_map=uncertainty_map,
        )

    print(f'Phase 2 MedSAM done: {len(samples)} files → {args.output_dir}')


if __name__ == '__main__':
    run()
