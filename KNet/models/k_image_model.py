"""
K_image_model.py

This file defines joint k-space and image-domain segmentation models.
It includes gating and attention mechanisms to fuse k-space features
with image features, enabling multi-modal learning for improved
segmentation performance.

For both fully sampling and undersampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.main_utils import dice_loss
from models.modules import KspaceRNNEncoder, CoarseHead, UNet


class GateFuse(nn.Module):
    """
    Gated fusion module for combining coarse and image features.

    This module learns a pixel-wise gating map to adaptively fuse
    two input feature maps.
    """
    def __init__(self, in_ch=2, mid=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid()
        )

    def forward(self, coarse, img):
        g = self.net(torch.cat([coarse, img], 1))
        return g * coarse + (1.0 - g) * img, g


# class KImgSegModel(nn.Module):
#     def __init__(self, k_in_ch: int=2, num_classes: int = 4,
#                  nf: int = 64, unet_base: int = 16, use_in: bool = True):
#         super().__init__()
#         self.k_in_ch = k_in_ch  # 2*C
#         self.enc    = KspaceRNNEncoder(in_ch=k_in_ch, nf=nf, fuse="concat")
#         self.coarse = CoarseHead(self.enc.out_ch)            # -> [B,1,H,W]
#         self.refiner= UNet(in_ch=2, out_ch=num_classes,      # coarse+img
#                            base=unet_base, use_in=use_in)
#         self.recon_head = nn.Sequential(
#             nn.Conv2d(self.enc.out_ch, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 1, 3, padding=1)
#         )
#         self.apply(self._init_weights)
#
#     @staticmethod
#     def _init_weights(m):
#         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#             nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#             if m.bias is not None: nn.init.zeros_(m.bias)
#
#     def forward(self, x):  # x: [B, 2*C+1, H, W]
#         k_ri = x[:, :self.k_in_ch, ...]
#         img  = x[:, self.k_in_ch:self.k_in_ch+1, ...]    # [B,1,H,W]
#         feat   = self.enc(k_ri)
#         coarse = self.coarse(feat)
#         self.img_rec_logits = self.recon_head(feat)
#         ref_in = torch.cat([coarse, img], dim=1)
#         return self.refiner(ref_in)                      # logits
#
#     def loss(self, logits, target, x_for_recon=None):
#         ce = F.cross_entropy(logits, target)
#         dice = dice_loss(logits, target)
#         total = ce + dice
#         return {"ce": ce, "dice": dice, "total": total}


class KImgSegModel2(nn.Module):
    """
        Multi-modal segmentation model combining k-space and image features.

        This model encodes k-space data with a recurrent encoder, generates
        a coarse reconstruction, and fuses it with image features using
        channel-wise gating and pixel attention. A U-Net refines the fused
        representation to produce segmentation outputs.

        Parameters
        ----------
        k_in_ch : int, default=2
            Number of k-space input channels (e.g., real + imaginary).
        num_classes : int, default=4
            Number of segmentation classes.
        nf : int, default=64
            Feature dimension for the k-space encoder.
        unet_base : int, default=24
            Base number of feature channels for the U-Net.
        use_in : bool, default=True
            Whether to use Instance Normalization in the U-Net.
        drop_img_p : float, default=0.0
            Probability of dropping the image channel during training
            to encourage reliance on k-space features.

        Attributes
        ----------
        enc : nn.Module
            K-space recurrent encoder.
        coarse : nn.Module
            Coarse reconstruction head producing [B,1,H,W].
        se_gate : nn.Module
            Squeeze-and-excitation gate for channel-wise modulation.
        pix_att : nn.Module
            Pixel-level attention mechanism.
        refiner : nn.Module
            U-Net segmentation head refining fused inputs.

        Loss function
        ----------
        loss(logits: torch.Tensor, target: torch.Tensor, *_) -> dict
            Compute segmentation losses.
            Returns a dictionary with:
            - "ce": Cross-entropy loss.
            - "dice": Dice loss.
            - "total": Sum of cross-entropy and Dice losses.
    """
    def __init__(self, k_in_ch: int = 2, num_classes: int = 4,
                 nf: int = 64, unet_base: int = 24, use_in: bool = True,
                 drop_img_p: float = 0.0):
        super().__init__()
        self.k_in_ch = int(k_in_ch)
        self.drop_img_p = float(drop_img_p)

        # k-space encoder
        self.enc = KspaceRNNEncoder(in_ch=self.k_in_ch, nf=nf, fuse="concat")
        self.coarse = CoarseHead(self.enc.out_ch)

        # fuse modalities (attention mechanism, gating)
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.enc.out_ch, max(16, self.enc.out_ch//8), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(16, self.enc.out_ch//8), 1, 1), nn.Sigmoid()
        )
        # pixel attention, shape:[Ｂ,1,H,W]
        self.pix_att = nn.Sequential(
            nn.Conv2d(self.enc.out_ch, 1, 1), nn.Sigmoid()
        )
        # UNet with 2 channel input：k-space coarse + image
        self.refiner = UNet(in_ch=2, out_ch=num_classes, base=unet_base, use_in=use_in)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.k_in_ch + 1, \
            f"Expect [B,{self.k_in_ch+1},H,W], got {tuple(x.shape)}"

        k_ri = x[:, :self.k_in_ch, ...]
        img = x[:, self.k_in_ch:self.k_in_ch+1, ...]
        feat = self.enc(k_ri)
        coarse = self.coarse(feat)

        g = self.se_gate(feat)
        A = self.pix_att(feat)

        # k-space gate
        coarse_mod = g * coarse
        # pixel attention tuning images
        img_mod = (1.0 + A) * img

        ref_in = torch.cat([coarse_mod, img_mod], dim=1)
        return self.refiner(ref_in)

    def loss(self, logits, target, *_):
        ce = F.cross_entropy(logits, target)
        dice = dice_loss(logits, target)
        total = ce + dice
        return {"ce": ce, "dice": dice, "total": total}
