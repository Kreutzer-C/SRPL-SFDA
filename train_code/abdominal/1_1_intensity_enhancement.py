# -*- coding: utf-8 -*-
"""
Phase 1: Test-Time Tri-branch Intensity Enhancement + Ensemble Pseudo-Label Generation.
Adapted from the original SRPL-SFDA fetal_brain pipeline (1_1~1_4) for ABDOMINAL npz format.

Three intensity transformations (identical to original SRPL-SFDA):
  1) Histogram equalization (per-slice, cv2.equalizeHist)
  2) Gamma correction γ=0.6 (rD: "reduce Darkness")
  3) Gamma correction optimized to match source statistics (rS: "reduce Style",
     target mean=0.5, target std=0.29, via Nelder-Mead)

For each target slice, the source model predicts on all 3 enhanced versions,
then the probability maps are averaged (ensemble) to produce the final pseudo-label.
Uncertainty is computed from the ensemble probability entropy.

Supports bidirectional adaptation:
  --source_domain BTCV    → target=CHAOST2 (CT→MR)
  --source_domain CHAOST2 → target=BTCV (MR→CT)
"""
import os, sys, argparse, numpy as np, torch, cv2, json
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.optimize import minimize

ROOT = '/opt/data/private/SRPL-SFDA'
sys.path.insert(0, ROOT)

from networks.net_factory import net_factory
from dataloaders.abdominal_dataset import NUM_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('--source_model', type=str, default=None,
                    help='Path to source model checkpoint. Default: results/{S}2{T}/source_model/best_model.pth')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output root dir. Default: data/ABDOMINAL/{S}2{T}/enhanced_pl')
parser.add_argument('--source_domain', type=str, default='BTCV', choices=['BTCV', 'CHAOST2'],
                    help='Source domain key. BTCV=CT→MR, CHAOST2=MR→CT')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()

SOURCE_KEY = args.source_domain
TARGET_KEY = 'CHAOST2' if SOURCE_KEY == 'BTCV' else 'BTCV'
DIRECTION = f'{SOURCE_KEY}2{TARGET_KEY}'

if args.source_model is None:
    args.source_model = os.path.join(ROOT, 'results', DIRECTION, 'source_model', 'best_model.pth')
if args.output_dir is None:
    args.output_dir = os.path.join(ROOT, 'data', 'ABDOMINAL', DIRECTION, 'enhanced_pl')


def _collect_target_slices(split_name, target_key):
    """Collect all target-domain npz slices for the given split."""
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
    """Load npz, normalize, resize to patch_size. Returns (img_01, original_img)."""
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
# Intensity transformation 1: Histogram Equalization (exact copy of original 1_1)
# ---------------------------------------------------------------------------
def apply_histogram_equalization(img_01):
    """Per-slice histogram equalization using cv2.equalizeHist."""
    img_uint8 = (img_01 * 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_uint8)
    return equalized.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Intensity transformation 2: Gamma Correction γ=0.6 (exact copy of original 1_2)
# ---------------------------------------------------------------------------
def gamma_correction(image, gamma):
    return np.power(image + 1e-10, gamma)


def apply_gamma_fixed(img_01, gamma=0.6):
    """Gamma correction with fixed gamma=0.6. Processes non-zero pixels only."""
    mask = img_01 > 0
    if mask.sum() == 0:
        return img_01.copy()
    foreground = img_01[mask]
    fg_norm = (foreground - foreground.min()) / (foreground.max() - foreground.min() + 1e-10)
    corrected = gamma_correction(fg_norm, gamma)
    result = np.zeros_like(img_01)
    result[mask] = corrected
    return result


# ---------------------------------------------------------------------------
# Intensity transformation 3: Gamma Correction matching source stats (exact copy of original 1_3)
# ---------------------------------------------------------------------------
def _optimize_gamma(processed_data, target_mean, target_std):
    def objective(gamma, image, tgt_mean, tgt_std):
        corrected = gamma_correction(image, gamma)
        return abs((corrected.mean() - tgt_mean) + (corrected.std() - tgt_std))
    result = minimize(objective, 1.0, args=(processed_data, target_mean, target_std),
                      method='Nelder-Mead')
    return result.x[0]


def apply_gamma_matched(img_01, target_mean=0.5, target_std=0.29):
    """Gamma correction optimized to match target intensity statistics."""
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
# Main ensemble pseudo-label generation
# ---------------------------------------------------------------------------
def generate():
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    # Load source model
    model = net_factory(net_type='unet2d', in_chns=1, class_num=NUM_CLASSES)
    model.load_state_dict(torch.load(args.source_model, map_location='cuda:0'))
    model.cuda()
    model.eval()

    samples = _collect_target_slices(args.split, TARGET_KEY)
    print(f'Phase 1 [{DIRECTION} | {args.split}]: {len(samples)} slices from {TARGET_KEY}')
    print(f'  Transformations: (1) Histogram EQ  (2) Gamma=0.6  (3) Gamma→mean=0.5,std=0.29')
    print(f'  Source model: {args.source_model}')
    print(f'  Output: {out_dir}')

    for npz_path, fname in tqdm(samples, desc=f'{TARGET_KEY} {args.split}'):
        img_01 = load_and_preprocess(npz_path, tuple(args.patch_size))

        # Apply 3 intensity transformations
        img_eq = apply_histogram_equalization(img_01)
        img_rd = apply_gamma_fixed(img_01, gamma=0.6)
        img_rs = apply_gamma_matched(img_01, target_mean=0.5, target_std=0.29)

        # Stack all 3 transformed images → [3, H, W] tensor for batch inference
        imgs = np.stack([img_eq, img_rd, img_rs], axis=0)  # [3, H, W]
        imgs_tensor = torch.from_numpy(imgs).unsqueeze(1).float().cuda()  # [3, 1, H, W]

        with torch.no_grad():
            logits = model(imgs_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # [3, C, H, W]

        # Ensemble: average probability maps across the 3 branches
        avg_probs = probs.mean(axis=0)  # [C, H, W]

        # Pseudo-label: argmax of ensemble probabilities
        pseudo_label = np.argmax(avg_probs, axis=0).astype(np.uint8)  # [H, W]

        # Uncertainty: normalized entropy of ensemble probabilities
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-6), axis=0)  # [H, W]
        uncertainty_map = (entropy / np.log(NUM_CLASSES)).astype(np.float32)

        np.savez_compressed(
            os.path.join(out_dir, f'{fname}.npz'),
            pseudo_label=pseudo_label,
            uncertainty_map=uncertainty_map,
        )

    print(f'Phase 1 done: {len(samples)} files → {out_dir}')


if __name__ == '__main__':
    generate()
