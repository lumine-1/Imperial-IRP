"""
image_model.py

This file defines a simple image-domain segmentation model.
It wraps the U-Net that is the same as other models for semantic segmentation of
reconstructed images, using cross-entropy and Dice loss
as the training objectives.

For both fully sampling and undersampling
"""

from __future__ import annotations

import torch.nn as nn
from models.modules import UNet
from utils.main_utils import dice_loss
import torch.nn.functional as F


class ImageSegModel(nn.Module):
    """
    A lightweight U-Net based segmentation model.
    This model wraps a U-Net backbone for semantic segmentation tasks.

    Parameters
    ----------
    n_classes : int, default=4
        Number of output classes for segmentation.
    base : int, default=24
        Base number of feature channels in the U-Net.
    use_in : bool, default=True
        Whether to use Instance Normalization layers in the U-Net.
    """
    def __init__(
        self,
        n_classes: int = 4,
        base: int = 24,
        use_in: bool = True,
    ):
        super().__init__()
        # use Unet in the modules directly
        self.net = UNet(
            in_ch=1, out_ch=n_classes, base=base, use_in=use_in
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def loss(self, logits, target, *_):
        ce = F.cross_entropy(logits, target)
        dice = dice_loss(logits, target)
        # use ce + dice loss as loss function
        total = ce + dice
        return {"ce": ce, "dice": dice, "total": total}


