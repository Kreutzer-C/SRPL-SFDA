# -*- coding: utf-8 -*-
"""
Evaluate model on ABDOMINAL test set. Reports per-class and mean Dice/HD95/ASSD.

Supports bidirectional evaluation via --source_domain.
"""
import os, sys, argparse, logging
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric
from scipy.ndimage import zoom

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


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        try:
            hd95 = metric.binary.hd95(pred, gt)
        except:
            hd95 = 373.0  # sentinel for failed HD95
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
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
    model.eval()

    dataset = AbdominalDataset(split=args.split, domain=args.domain, patch_size=tuple(args.patch_size),
                               source_domain_key=SOURCE_KEY)
    direction = f'{SOURCE_KEY}2{TARGET_KEY}'
    logging.info(f'Evaluating [{direction} | {args.domain}/{args.split}]: {len(dataset)} slices')

    metrics = {c: {'dice': [], 'hd95': [], 'assd': []} for c in range(1, NUM_CLASSES)}

    for i in tqdm(range(len(dataset)), ncols=70):
        sample = dataset[i]
        img = sample['image'].unsqueeze(0).cuda()
        gt = sample['gt'].numpy()

        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1)[0].cpu().numpy()

        for c in range(1, NUM_CLASSES):
            d, h, a = calculate_metric_percase(pred == c, gt == c)
            metrics[c]['dice'].append(d)
            metrics[c]['hd95'].append(h)
            metrics[c]['assd'].append(a)

    # Report
    all_dice = []
    print(f'\n{"Class":<20} {"Dice":>8} {"HD95":>8} {"ASSD":>8}')
    print('-' * 50)
    for c in range(1, NUM_CLASSES):
        d = np.mean(metrics[c]['dice'])
        h = np.mean(metrics[c]['hd95'])
        a = np.mean(metrics[c]['assd'])
        all_dice.append(d)
        print(f'{CLASS_NAMES[c]:<20} {d:>8.4f} {h:>8.2f} {a:>8.2f}')

    print('-' * 50)
    print(f'{"Mean":<20} {np.mean(all_dice):>8.4f}')

    return np.mean(all_dice)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    mean_dice = evaluate()
    logging.info(f'Final Mean Dice: {mean_dice:.4f}')
