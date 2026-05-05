# -*- coding: utf-8 -*-
"""
Evaluate MemProp-trained source models on both source and target domains.

Usage:
  python eval_memprop_source.py \
    --model_path /path/to/checkpoint.pth \
    --source_domain BTCV \
    --domain source  # or target or both

Supports both raw state_dict and MemProp's {'model_state_dict': ...} checkpoint format.
"""
import os, sys, argparse, numpy as np, torch
from tqdm import tqdm
from medpy import metric

ROOT = '/opt/data/private/SRPL-SFDA'
sys.path.insert(0, ROOT)

from networks.net_factory import net_factory
from dataloaders.abdominal_dataset import AbdominalDataset, NUM_CLASSES, CLASS_NAMES


def load_ckpt(model, path):
    """Load checkpoint, handling both raw state_dict and MemProp wrapper."""
    ckpt = torch.load(path, map_location='cuda:0')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model


def calc_metric(pred, gt):
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


def evaluate(model_path, source_domain, eval_domain, split='test'):
    model = net_factory(net_type='unet2d', in_chns=1, class_num=NUM_CLASSES)
    model = load_ckpt(model, model_path)
    model.eval()

    dataset = AbdominalDataset(split=split, domain=eval_domain, patch_size=(256, 256),
                               source_domain_key=source_domain)
    name = os.path.basename(model_path)
    print('\n========================================')
    print('Model: %s' % name)
    print('Source=%s | Domain=%s | Split=%s | Slices=%d' % (source_domain, eval_domain, split, len(dataset)))
    print('========================================')

    metrics = {}
    for c in range(1, NUM_CLASSES):
        metrics[c] = {'dice': [], 'hd95': [], 'assd': []}

    for i in tqdm(range(len(dataset)), ncols=70, desc='%s/%s' % (source_domain, eval_domain)):
        sample = dataset[i]
        img = sample['image'].unsqueeze(0).cuda()
        gt = sample['gt'].numpy()
        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1)[0].cpu().numpy()
        for c in range(1, NUM_CLASSES):
            d, h, a = calc_metric(pred == c, gt == c)
            metrics[c]['dice'].append(d)
            metrics[c]['hd95'].append(h)
            metrics[c]['assd'].append(a)

    all_dice = []
    header = '%-20s %8s %8s %8s' % ('Class', 'Dice', 'HD95', 'ASSD')
    print(header)
    print('-' * 50)
    for c in range(1, NUM_CLASSES):
        d = np.mean(metrics[c]['dice'])
        h = np.mean(metrics[c]['hd95'])
        a = np.mean(metrics[c]['assd'])
        all_dice.append(d)
        print('%-20s %8.4f %8.2f %8.2f' % (CLASS_NAMES[c], d, h, a))
    print('-' * 50)
    mean_d = np.mean(all_dice)
    print('%-20s %8.4f' % ('Mean', mean_d))
    return mean_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MemProp source models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--source_domain', type=str, required=True, choices=['BTCV', 'CHAOST2'])
    parser.add_argument('--domain', type=str, default='both', choices=['source', 'target', 'both'],
                        help='Which domain to evaluate on (default: both)')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    results = {}

    if args.domain in ('source', 'both'):
        key = '%s_src_on_%s' % (args.source_domain, args.source_domain)
        results[key] = evaluate(args.model_path, args.source_domain, 'source', args.split)

    if args.domain in ('target', 'both'):
        target_key = 'CHAOST2' if args.source_domain == 'BTCV' else 'BTCV'
        key = '%s_src_on_%s' % (args.source_domain, target_key)
        results[key] = evaluate(args.model_path, args.source_domain, 'target', args.split)

    if len(results) > 1:
        print('\n' + '=' * 60)
        print('SUMMARY')
        print('=' * 60)
        for k, v in results.items():
            print('  %-30s Mean Dice = %.4f' % (k, v))
