# -*- coding: utf-8 -*-
"""
Stage 3 (CHAOS): Reliable Pseudo-Label (RPL) supervised fine-tuning.
Adapts 3_train_RPL_selectRPL_fine_tune.py for 5-class CHAOS with NPZ data.

Usage:
  python 3_train_RPL_selectRPL_fine_tune.py \
      --data_dir /path/to/ABDOMINAL/processed_DDFP \
      --domain CHAOST2 \
      --source_model /path/to/source_model.pth \
      --exp chaos_rpl_ft
"""
from __future__ import print_function, division
import argparse
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from scipy.ndimage.interpolation import zoom
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from dataloaders.chaos_dataset_RPL_selectRPL_UMviaEntropy import (
    ChaosSliceDataset, ChaosVolDataset, TrainToTensor, load_metadata,
)
from networks.net_factory import net_factory
from utils.losses import WeightedCrossEntropyLoss, WeightedDiceLoss
from utils.val_2D import test_single_volume

# CHAOS organ names for logging
ORGAN_NAMES = ["Liver", "R.Kidney", "L.Kidney", "Spleen"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to processed_DDFP directory (contains metadata.json)")
parser.add_argument("--domain", type=str, default="CHAOST2")
parser.add_argument("--exp", type=str, default="chaos_rpl_ft")
parser.add_argument("--net", type=str, default="unet2d")
parser.add_argument("--num_classes", type=int, default=5)
parser.add_argument("--source_model", type=str, required=True)
parser.add_argument("--pl_subdir", type=str, default="slices_sam_pl",
                    help="Subdirectory containing SAM-refined pseudo-labels")
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--T_fix", type=float, default=0.5,
                    help="Reliability threshold multiplier (threshold = T_fix * ln2)")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--base_lr", type=float, default=1e-4)
parser.add_argument("--lr_gamma", type=float, default=0.5,
                    help="LR decay factor every 1000 iterations")
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()


def reliability_based_threshold(uncertainty_map, threshold):
    return (uncertainty_map > threshold).float()


def train(args, snapshot_path, metadata):
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    T_fix = args.T_fix

    model = net_factory(net_type=args.net, in_chns=1, class_num=num_classes)
    model.load_state_dict(torch.load(args.source_model, map_location="cuda:0"))

    db_train = ChaosSliceDataset(
        base_dir=args.data_dir,
        domain=args.domain,
        metadata=metadata,
        split="train",
        pl_subdir=args.pl_subdir,
        transform=transforms.Compose([TrainToTensor(args.patch_size)]),
    )
    db_val = ChaosVolDataset(
        base_dir=args.data_dir,
        domain=args.domain,
        metadata=metadata,
        split="val",
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr,
                           betas=(0.5, 0.9), weight_decay=1e-5)
    ce_loss = WeightedCrossEntropyLoss()
    dice_loss = WeightedDiceLoss(num_classes)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    logging.info(f"{len(trainloader)} iterations per epoch")

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            uncertainty_map_batch = sampled_batch["uncertainty_map"].cuda()

            threshold = T_fix * 0.69315
            UM1_mask = reliability_based_threshold(uncertainty_map_batch, threshold)
            reliable_mask = (1.0 - UM1_mask).cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.long(), reliable_mask)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1),
                                  weight_map=reliable_mask)
            loss = 0.5 * (loss_ce + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (args.lr_gamma) ** (iter_num // 1000)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_

            iter_num += 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/loss", loss.item(), iter_num)
            logging.info(
                f"iter {iter_num}: loss={loss.item():.4f} "
                f"ce={loss_ce.item():.4f} dice={loss_dice.item():.4f}"
            )

            # ---- Validation ----
            if iter_num > 0 and iter_num % 50 == 0:
                model.eval()
                metric_sum = np.zeros((num_classes - 1, 3))  # (4, 3): dice/hd95/assd
                dice_per_vol = []

                for val_batch in valloader:
                    metric_i = test_single_volume(
                        val_batch["image"], val_batch["gt"],
                        model, classes=num_classes,
                    )
                    metric_sum += np.array(metric_i)
                    dice_per_vol.append(np.mean([m[0] for m in metric_i]))

                metric_mean = metric_sum / len(db_val)
                mean_dice = float(np.mean(metric_mean[:, 0]))

                for ci, name in enumerate(ORGAN_NAMES):
                    writer.add_scalar(f"val/{name}_dice", metric_mean[ci, 0], iter_num)
                    writer.add_scalar(f"val/{name}_hd95", metric_mean[ci, 1], iter_num)
                writer.add_scalar("val/mean_dice", mean_dice, iter_num)

                logging.info(
                    "Validation iter %d | mean_dice=%.4f | "
                    + " | ".join(f"{n}={metric_mean[i,0]:.4f}"
                                 for i, n in enumerate(ORGAN_NAMES)),
                    iter_num, mean_dice,
                )

                if mean_dice > best_performance:
                    best_performance = mean_dice
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path,
                                            f"iter_{iter_num}_dice_{mean_dice:.4f}.pth"))
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path,
                                            f"{args.net}_best_model.pth"))
                model.train()

            if iter_num % 500 == 0:
                torch.save(model.state_dict(),
                           os.path.join(snapshot_path, f"iter_{iter_num}.pth"))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    torch.save(model.state_dict(),
               os.path.join(snapshot_path, f"iter_{max_iterations + 1}.pth"))
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    metadata = load_metadata(args.data_dir)
    assert args.domain in metadata["domains"], \
        f"Domain '{args.domain}' not in metadata: {metadata['domains']}"

    snapshot_path = os.path.join(
        "../../results/chaos_rpl",
        args.exp, args.domain,
        f"iters{args.max_iterations}_lr{args.base_lr}_T{args.T_fix}",
    )
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    start = datetime.datetime.now()
    train(args, snapshot_path, metadata)
    elapsed = (datetime.datetime.now() - start).seconds / 3600
    print(f"Finished. Total time: {elapsed:.2f} h")
