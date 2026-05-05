# -*- coding: utf-8 -*-
"""
Evaluate model on ABDOMINAL test set. Reports per-class and mean Dice/HD95/ASSD.

Metrics are computed at the 3D volume level: all slices of a case are stacked,
then per-class metrics are computed on the full volume. Results are averaged
across volumes (not slices).

Supports bidirectional evaluation via --source_domain.
Compatible with both raw state_dict and MemProp checkpoint format
({'model_state_dict': ...}).
"""
import os, sys, argparse, logging
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric

ROOT = '/opt/data/private/SRPL-SFDA'
sys.path.insert(0, ROOT)

from networks.net_factory import net_factory
from dataloaders.abdominal_dataset import AbdominalDataset, NUM_CLASSES, CLASS_NAMES

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--source_domain', type=str, default='BTCV', choices=['BTCV', 'CHAOST2'],
                    help='Source domain key. BTCV=CT→MR, CHAOST2=MR→CT')
parser.add_argument('--domain', type=str, default='target', choices=['source', 'target'])
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
args = parser.parse_args()

SOURCE_KEY = args.source_domain
TARGET_KEY = 'CHAOST2' if SOURCE_KEY == 'BTCV' else 'BTCV'


def load_checkpoint(model, path):
    """Load checkpoint, handling both raw state_dict and MemProp wrapper format."""
    ckpt = torch.load(path, map_location='cuda:0')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        try:
            hd95 = metric.binary.hd95(pred, gt)
        except:
            hd95 = 373.0
        try:
            assd = metric.binary.asd(pred, gt)
        except:
            assd = 200.0
        return dice, hd95, assd
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    else:
        return 0.0, 373.0, 200.0


def evaluate():
    model = net_factory(net_type='unet2d', in_chns=1, class_num=NUM_CLASSES)
    model = load_checkpoint(model, args.model_path)
    model.eval()

    dataset = AbdominalDataset(split=args.split, domain=args.domain, patch_size=tuple(args.patch_size),
                               source_domain_key=SOURCE_KEY)
    direction = f'{SOURCE_KEY}2{TARGET_KEY}'
    logging.info(f'Evaluating [{direction} | {args.domain}/{args.split}]: {len(dataset)} slices')

    # Group slices by volume (case_id)
    case_slices = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        fname = sample['image_name']          # e.g. "vol_0001_001"
        case_id = fname.split('_')[1]          # "0001"
        slice_idx = int(fname.split('_')[3])   # 12 (format: vol_0001_slice_0012)
        case_slices[case_id].append((slice_idx, i))

    # Sort each case's slices by slice index
    for cid in case_slices:
        case_slices[cid].sort(key=lambda x: x[0])

    case_ids = sorted(case_slices.keys())

    # Per-class volume-level metrics
    vol_metrics = {c: {'dice': [], 'hd95': [], 'assd': []} for c in range(1, NUM_CLASSES)}

    for case_id in tqdm(case_ids, desc=f'[{direction}]', ncols=70):
        items = case_slices[case_id]
        pred_slices = []
        gt_slices = []

        for _, idx in items:
            sample = dataset[idx]
            img = sample['image'].unsqueeze(0).cuda()
            gt = sample['gt'].numpy()

            with torch.no_grad():
                out = model(img)
                pred = torch.argmax(torch.softmax(out, dim=1), dim=1)[0].cpu().numpy()

            pred_slices.append(pred)
            gt_slices.append(gt)

        pred_vol = np.stack(pred_slices, axis=0)  # [D, H, W]
        gt_vol = np.stack(gt_slices, axis=0)

        for c in range(1, NUM_CLASSES):
            d, h, a = calculate_metric_percase(pred_vol == c, gt_vol == c)
            vol_metrics[c]['dice'].append(d)
            vol_metrics[c]['hd95'].append(h)
            vol_metrics[c]['assd'].append(a)

    # Report
    all_dice = []
    print(f'\n{"Class":<20} {"Dice":>8} {"HD95":>8} {"ASSD":>8}')
    print('-' * 50)
    for c in range(1, NUM_CLASSES):
        d = np.mean(vol_metrics[c]['dice'])
        h = np.mean(vol_metrics[c]['hd95'])
        a = np.mean(vol_metrics[c]['assd'])
        all_dice.append(d)
        print(f'{CLASS_NAMES[c]:<20} {d:>8.4f} {h:>8.2f} {a:>8.2f}')

    print('-' * 50)
    mean_d = np.mean(all_dice)
    print(f'{"Mean":<20} {mean_d:>8.4f}')
    print(f'  (averaged over {len(case_ids)} volumes)')

    return mean_d


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    mean_dice = evaluate()
    logging.info(f'Final Mean Dice: {mean_dice:.4f}')
