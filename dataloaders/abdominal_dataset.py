# -*- coding: utf-8 -*-
import os, json
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

ROOT = '/opt/data/private/SRPL-SFDA'
DATASET_DIR = os.path.join(ROOT, 'datasets', 'datasets', 'ABDOMINAL', 'processed_DDFP')

with open(os.path.join(DATASET_DIR, 'metadata.json'), 'r') as f:
    META = json.load(f)

CLASS_NAMES = META['class_names']
NUM_CLASSES = META['num_classes']


def _get_cases(domain_key, split_name):
    splits = META['splits'][domain_key]
    if split_name in ('val', 'test'):
        return list(splits['test'])
    return list(splits['train'])


def _collect_slice_paths(domain_key, case_ids):
    paths = []
    slice_dir = os.path.join(DATASET_DIR, domain_key, 'slices')
    if not os.path.isdir(slice_dir):
        return paths
    for case_id in case_ids:
        pattern = f'vol_{case_id}_'
        for fname in sorted(os.listdir(slice_dir)):
            if fname.startswith(pattern) and fname.endswith('.npz'):
                paths.append((os.path.join(slice_dir, fname), fname.replace('.npz', '')))
    return paths


class AbdominalDataset(Dataset):
    """
    Args:
        domain: 'source' or 'target'
        source_domain_key: domain key for the source side (default 'BTCV').
                           The target side is automatically the other domain.
                           'BTCV' → source=BTCV, target=CHAOST2  (CT→MR)
                           'CHAOST2' → source=CHAOST2, target=BTCV  (MR→CT)
        pseudo_label_dir: path to pre-generated pseudo-label npz directory.
    """
    def __init__(self, base_dir=None, split='train', domain='source',
                 transform=None, patch_size=(256, 256),
                 pseudo_label_dir=None, source_domain_key='BTCV'):
        self.base_dir = base_dir or os.path.join(ROOT, 'data', 'ABDOMINAL')
        self.split = split
        self.domain = domain
        self.transform = transform
        self.patch_size = patch_size
        self.pseudo_label_dir = pseudo_label_dir

        self.source_key = source_domain_key
        self.target_key = 'CHAOST2' if source_domain_key == 'BTCV' else 'BTCV'
        domain_key = self.source_key if domain == 'source' else self.target_key
        case_ids = _get_cases(domain_key, split)
        self.samples = _collect_slice_paths(domain_key, case_ids)

        label_source = 'GT' if domain == 'source' else (
            f'PL dir={pseudo_label_dir}' if pseudo_label_dir else 'zeros')
        direction = f'{self.source_key}→{self.target_key}'
        print(f'AbdominalDataset [{direction} | {domain}/{split}]: {len(self.samples)} slices '
              f'from {len(case_ids)} cases, labels={label_source}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, fname = self.samples[idx]
        data = np.load(npz_path)
        img = data['img'].astype(np.float32)
        gt = data['label'].astype(np.int64)

        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        h, w = img.shape
        img = zoom(img, (self.patch_size[0] / h, self.patch_size[1] / w), order=1)
        gt = zoom(gt, (self.patch_size[0] / h, self.patch_size[1] / w), order=0)

        if self.domain == 'source':
            pseudo_label = gt.copy()
            uncertainty_map = np.zeros_like(img, dtype=np.float32)
        elif self.pseudo_label_dir and os.path.isdir(self.pseudo_label_dir):
            pl_path = os.path.join(self.pseudo_label_dir, f'{fname}.npz')
            if os.path.exists(pl_path):
                pl_data = np.load(pl_path)
                pseudo_label = pl_data['pseudo_label'].astype(np.int64)
                uncertainty_map = pl_data['uncertainty_map'].astype(np.float32)
            else:
                pseudo_label = np.zeros_like(gt)
                uncertainty_map = np.zeros_like(img, dtype=np.float32)
        else:
            pseudo_label = np.zeros_like(gt)
            uncertainty_map = np.zeros_like(img, dtype=np.float32)

        image_tensor = torch.from_numpy(img).unsqueeze(0).float()
        label_tensor = torch.from_numpy(pseudo_label).long()
        gt_tensor = torch.from_numpy(gt).long()
        um_tensor = torch.from_numpy(uncertainty_map).float()

        return {
            'image': image_tensor, 'label': label_tensor,
            'gt': gt_tensor, 'uncertainty_map': um_tensor,
            'image_name': fname, 'idx': idx,
        }
