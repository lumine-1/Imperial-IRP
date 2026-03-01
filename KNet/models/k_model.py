"""
K_model.py

This file defines the k-space only segmentation model.

It provides a configuration dataclass and the `KSegModel`,
which combines a recurrent k-space encoder, a coarse head,
a U-Net refiner, and an auxiliary reconstruction head for
self-supervised learning.

For both fully sampling and undersampling
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import KspaceRNNEncoder, CoarseHead, UNet
from utils.main_utils import dice_loss


# Model settings
@dataclass
class ModelCfg:
    in_ch: int = 2
    nf: int = 64
    num_classes: int = 4
    unet_base: int = 24
    use_in: bool = True


# Final model - full sample/Undersample K
class KSegModel(nn.Module):
    """
    A K-space segmentation model, only use K-space.

    This model combines a recurrent K-space encoder, a coarse prediction head,
    a U-Net refiner, and an auxiliary image reconstruction head. It supports
    segmentation with additional supervision from image reconstruction.

    Parameters
    ----------
    cfg : ModelCfg
        Configuration object specifying input channels, number of filters,
        U-Net base channels, normalization usage, and number of segmentation classes.

    Attributes
    ----------
    enc : nn.Module
        K-space encoder using recurrent layers to extract features.
    coarse : nn.Module
        Coarse segmentation head operating on encoder features.
    refiner : nn.Module
        U-Net refiner for final segmentation logits.
    recon_head : nn.Sequential
        Auxiliary image reconstruction head for self-supervised supervision.
    img_rec_logits : torch.Tensor
        Reconstructed image logits (set during forward pass).

    Loss function
    -------
    loss(logits: torch.Tensor, mask: torch.Tensor, kspace: torch.Tensor) -> dict
        Compute combined loss terms:
        - "ce": Cross-entropy loss for segmentation.
        - "dice": Dice loss for segmentation.
        - "recon": L1 reconstruction loss between predicted and ground-truth images.
        - "total": Weighted sum of segmentation and reconstruction losses.
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.enc = KspaceRNNEncoder(cfg.in_ch, cfg.nf, "concat")
        self.coarse = CoarseHead(self.enc.out_ch)
        self.refiner = UNet(1, cfg.num_classes, base=cfg.unet_base, use_in=cfg.use_in)
        self.recon_head = nn.Sequential(
            nn.Conv2d(self.enc.out_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, k_ri):
        # k space encoder
        feat = self.enc(k_ri)
        # recon head
        self.img_rec_logits = self.recon_head(feat)
        # coarse head
        coarse = self.coarse(feat)
        # Unet refiner
        return self.refiner(coarse)

    def loss(self, logits, mask, kspace):
        ce = F.cross_entropy(logits, mask)
        dice = dice_loss(logits, mask)

        # supervision loss
        k_cpx = torch.complex(kspace[:, 0], kspace[:, 1])
        with torch.no_grad():
            img_cpx = torch.fft.ifft2(k_cpx, norm='ortho')
            img_mag_gt = torch.abs(img_cpx).unsqueeze(1)
        recon_loss = F.l1_loss(self.img_rec_logits, img_mag_gt)
        # loss = ce + dice loss + supervision loss
        # when training using the fully sampled data, this number should be 3
        total = ce + dice + 0 * recon_loss

        return {
            "ce": ce,
            "dice": dice,
            "recon": recon_loss,
            "total": total
        }




