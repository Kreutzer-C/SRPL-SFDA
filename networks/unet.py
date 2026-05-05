# -*- coding: utf-8 -*-
"""
UNet upgraded to MemProp-SFDA alignment.

Compatible with:
  - MemProp-SFDA checkpoints (identical architecture & state-dict keys)
  - SRPL-SFDA existing callers: UNet(in_chns=..., class_num=...)

Based on MemProp-SFDA /networks/unet_modeling.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # handle non-power-of-2 sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet aligned with MemProp-SFDA.

    Parameters (MemProp-style):
        n_channels: input channels (default 1)
        n_classes: output classes (default 5)
        first_channels: base channel count (default 64 → [64,128,256,512,1024])
        only_feature: return decoder features instead of logits
        only_logits: return logits only (True); if False returns (features, logits)
        bilinear: use bilinear upsampling instead of ConvTranspose2d

    Parameters (SRPL-legacy, also accepted):
        in_chns: alias for n_channels
        class_num: alias for n_classes
    """

    def __init__(self, in_chns=None, class_num=None,
                 n_channels=None, n_classes=None,
                 first_channels=64, only_feature=False,
                 only_logits=True, bilinear=False):
        super(UNet, self).__init__()

        # Resolve legacy vs new parameter names
        n_channels = n_channels if n_channels is not None else (in_chns if in_chns is not None else 1)
        n_classes = n_classes if n_classes is not None else (class_num if class_num is not None else 5)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.only_feature = only_feature
        self.only_logits = only_logits
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, first_channels)
        self.down1 = Down(first_channels, first_channels * 2)
        self.down2 = Down(first_channels * 2, first_channels * 4)
        self.down3 = Down(first_channels * 4, first_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(first_channels * 8, first_channels * 16 // factor)
        self.up1 = Up(first_channels * 16, first_channels * 8 // factor, bilinear)
        self.up2 = Up(first_channels * 8, first_channels * 4 // factor, bilinear)
        self.up3 = Up(first_channels * 4, first_channels * 2 // factor, bilinear)
        self.up4 = Up(first_channels * 2, first_channels, bilinear)
        if not self.only_feature:
            self.outc = OutConv(first_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.only_feature:
            return x
        elif self.only_logits:
            return self.outc(x)
        else:
            return x, self.outc(x)


def build_unet(config_path, img_size=256, num_classes=5):
    """Build UNet from a JSON config file (MemProp-compatible)."""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return UNet(n_channels=config.get('in_channels', 1),
                n_classes=num_classes,
                first_channels=config.get('first_channels', 64),
                only_feature=config.get('only_feature', False),
                only_logits=config.get('only_logits', True),
                bilinear=config.get('bilinear', False))


if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=5)
    print(model)
    print(f">>> Parameters: {sum(p.numel() for p in model.parameters()):,}")
