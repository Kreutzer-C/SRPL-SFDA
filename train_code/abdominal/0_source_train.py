# -*- coding: utf-8 -*-
"""Train source model on the source domain with supervised UNet (5-class).

Supports bidirectional training:
  --source_domain BTCV    → trains on BTCV (CT), adapt to CHAOST2 (MR)
  --source_domain CHAOST2 → trains on CHAOST2 (MR), adapt to BTCV (CT)
"""
import os, sys, random, logging, argparse, numpy as np
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

ROOT = '/opt/data/private/SRPL-SFDA'
sys.path.insert(0, ROOT)

from networks.net_factory import net_factory
from utils.losses import DiceCeLoss
from dataloaders.abdominal_dataset import AbdominalDataset, NUM_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('--source_domain', type=str, default='BTCV', choices=['BTCV', 'CHAOST2'],
                    help='Source domain key. BTCV=CT→MR, CHAOST2=MR→CT')
parser.add_argument('--exp', type=str, default=None,
                    help='Experiment name. Default: results/{S}2{T}/source_model/')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--save_interval', type=int, default=10)
args = parser.parse_args()

SOURCE_KEY = args.source_domain
TARGET_KEY = 'CHAOST2' if SOURCE_KEY == 'BTCV' else 'BTCV'
DIRECTION = f'{SOURCE_KEY}2{TARGET_KEY}'

if args.exp is None:
    args.exp = os.path.join(DIRECTION, 'source_model')


def validate(model, valloader):
    model.eval()
    dices = []
    with torch.no_grad():
        for batch in valloader:
            img = batch['image'].cuda()
            gt = batch['gt'].cuda()
            out = model(img)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
            for c in range(1, NUM_CLASSES):
                pred_c = (pred == c).float()
                gt_c = (gt == c).float()
                inter = (pred_c * gt_c).sum()
                union = pred_c.sum() + gt_c.sum()
                if union > 0:
                    dices.append((2 * inter / union).item())
    return np.mean(dices) if dices else 0.0


def train():
    snapshot_path = os.path.join(ROOT, 'results', args.exp)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=snapshot_path + '/log.txt', level=logging.INFO,
        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model = net_factory(net_type='unet2d', in_chns=1, class_num=NUM_CLASSES)
    logging.info(f'Model: UNet2D, classes={NUM_CLASSES}, direction={DIRECTION}')

    db_train = AbdominalDataset(split='train', domain='source', patch_size=tuple(args.patch_size),
                                source_domain_key=SOURCE_KEY)
    db_val = AbdominalDataset(split='val', domain='source', patch_size=tuple(args.patch_size),
                              source_domain_key=SOURCE_KEY)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.5, 0.9), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=args.lr_gamma)
    criterion = DiceCeLoss(num_classes=NUM_CLASSES, alpha=0.5)

    writer = SummaryWriter(snapshot_path + '/log')
    best_dice = 0.0
    iter_num = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(trainloader, desc=f'Epoch {epoch}/{args.max_epochs}', ncols=70):
            images = batch['image'].cuda()
            labels = batch['gt'].cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_num += 1
            epoch_loss += loss.item()
            writer.add_scalar('train/loss', loss.item(), iter_num)

        avg_loss = epoch_loss / len(trainloader)
        mean_dice = validate(model, valloader)
        logging.info(f'Epoch {epoch}: loss={avg_loss:.4f}, val_dice={mean_dice:.4f}')
        writer.add_scalar('val/mean_dice', mean_dice, iter_num)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))

        if epoch % args.save_interval == 0:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch}_dice_{mean_dice:.4f}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f'Saved: {save_path}')

        writer.flush()

    torch.save(model.state_dict(), os.path.join(snapshot_path, 'final_model.pth'))
    logging.info(f'Training finished. Best Dice: {best_dice:.4f}')
    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    train()
