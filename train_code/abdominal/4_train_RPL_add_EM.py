# -*- coding: utf-8 -*-
"""
SFDA Step 4: Reliability-Aware Pseudo-Label Supervision with Entropy Minimization.
Full SRPL-SFDA: L = L_sup(reliable) + lambda * L_EM(unreliable)

Supports bidirectional adaptation via --source_domain.
"""
import os, sys, random, logging, argparse
import numpy as np
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

ROOT = '/opt/data/private/SRPL-SFDA'
sys.path.insert(0, ROOT)

from networks.net_factory import net_factory
from utils.losses import WeightedDiceLoss, WeightedCrossEntropyLoss, WeightedEMLoss
from dataloaders.abdominal_dataset import AbdominalDataset, NUM_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('--source_model', type=str, required=True)
parser.add_argument('--source_domain', type=str, default='BTCV', choices=['BTCV', 'CHAOST2'],
                    help='Source domain key. BTCV=CT→MR, CHAOST2=MR→CT')
parser.add_argument('--exp', type=str, default='SFDA_step4_EM')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_iterations', type=int, default=3000)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=0.6)
parser.add_argument('--lameta_fix', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument("--pseudo_label_dir", type=str, default=None, help="path to ensemble pseudo-labels")
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
parser.add_argument('--save_interval', type=int, default=5,
                    help='save checkpoint every N epochs')
args = parser.parse_args()

SOURCE_KEY = args.source_domain
TARGET_KEY = 'CHAOST2' if SOURCE_KEY == 'BTCV' else 'BTCV'
DIRECTION = f'{SOURCE_KEY}2{TARGET_KEY}'


def validate(model, valloader):
    model.eval()
    dices_all = []
    with torch.no_grad():
        for vbatch in valloader:
            img = vbatch['image'].cuda()
            gt = vbatch['gt'].cuda()
            out = model(img)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
            dice_per_class = []
            for c in range(1, NUM_CLASSES):
                p = (pred == c).float()
                g = (gt == c).float()
                inter = (p * g).sum()
                denom = p.sum() + g.sum()
                dice_per_class.append((2 * inter / denom).item() if denom > 0 else 0.0)
            dices_all.append(dice_per_class)
    model.train()
    return np.mean(dices_all), np.mean(dices_all, axis=0)


def train():
    snapshot_path = os.path.join(ROOT, 'results', args.exp)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=snapshot_path + '/log.txt', level=logging.INFO,
        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'Direction: {DIRECTION}')
    logging.info(str(args))

    model = net_factory(net_type='unet2d', in_chns=1, class_num=NUM_CLASSES)
    model.load_state_dict(torch.load(args.source_model, map_location='cuda:0'))

    db_train = AbdominalDataset(split="train", domain="target", patch_size=tuple(args.patch_size),
                                pseudo_label_dir=args.pseudo_label_dir, source_domain_key=SOURCE_KEY)
    db_val = AbdominalDataset(split="val", domain="target", patch_size=tuple(args.patch_size),
                              pseudo_label_dir=args.pseudo_label_dir, source_domain_key=SOURCE_KEY)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.5, 0.9), weight_decay=1e-5)
    ce_loss = WeightedCrossEntropyLoss()
    dice_loss = WeightedDiceLoss(NUM_CLASSES)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_performance = 0.0
    final_threshold = args.threshold * 0.69315

    for epoch_num in range(1, max_epoch + 1):
        for batch in tqdm(trainloader, desc=f'Epoch {epoch_num}/{max_epoch}', ncols=70):
            images = batch['image'].cuda()
            labels = batch['label'].cuda()
            um = batch['uncertainty_map'].cuda()

            unreliable_mask = (um > final_threshold).float()
            reliable_mask = 1.0 - unreliable_mask

            outputs = model(images)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, labels.long(), reliable_mask)
            loss_dice = dice_loss(outputs_soft, labels.unsqueeze(1), weight_map=reliable_mask)
            sup_loss = 0.5 * (loss_dice + loss_ce)
            em_loss = WeightedEMLoss(outputs_soft, unreliable_mask)
            loss = sup_loss + args.lameta_fix * em_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (args.lr_gamma ** (iter_num // 1000))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            iter_num += 1
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/em_loss', em_loss.item(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)

            if iter_num >= args.max_iterations:
                break

        # End of epoch: validate and save
        mean_dice, class_dices = validate(model, valloader)
        writer.add_scalar('val/mean_dice', mean_dice, iter_num)
        for ci in range(NUM_CLASSES - 1):
            writer.add_scalar(f'val/class_{ci+1}_dice', class_dices[ci], iter_num)

        if mean_dice > best_performance:
            best_performance = mean_dice
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))

        if epoch_num % args.save_interval == 0:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_dice_{mean_dice:.4f}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f'Saved: {save_path}')

        logging.info(f'Epoch {epoch_num}: dice={mean_dice:.4f}, loss={loss.item():.4f}, '
                     f'sup={sup_loss.item():.4f}, em={em_loss.item():.4f}')

        if iter_num >= args.max_iterations:
            break

    torch.save(model.state_dict(), os.path.join(snapshot_path, 'final_model.pth'))
    logging.info(f'Training finished. Best Dice: {best_performance:.4f}')
    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    train()
