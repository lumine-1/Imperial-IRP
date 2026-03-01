"""
module.py

This file defines core neural network building blocks for k-space and image-based
learning tasks. It includes GRU-based sweep encoders, convolutional blocks,
and a U-Net implementation that serve as reusable components for segmentation
and reconstruction models.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class SweepGRU_2(nn.Module):
    """
    Bidirectional GRU sweeper module for spatial feature extraction.

    This module applies a stack of bidirectional GRU layers along a chosen
    spatial axis of the input feature map. It is useful for modeling
    long-range dependencies in k-space.

    Parameters
    ----------
    in_ch : int
        Number of input feature channels.
    hidden : int
        Hidden size of the GRU layers.
    axis : int
        Axis along which to sweep (-1 for width, -2 for height).
    num_layers : int, default=3
        Number of stacked GRU layers.
    residual : bool, default=True
        Whether to add residual connections between input and output
        when dimensions match.

    Shape
    -----
    Input:  (B, C, H, W)
    Output: (B, 2*hidden, H, W)
    """
    def __init__(self, in_ch, hidden, axis, num_layers: int = 3, residual: bool = True):
        super().__init__()
        assert axis in (-1, -2)
        self.axis, self.residual = axis, residual
        self.inter_drop = nn.Identity()

        self.layers = nn.ModuleList([
            nn.GRU(
                input_size=in_ch if i == 0 else hidden * 2,
                hidden_size=hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            for i in range(num_layers)
        ])
        for gru in self.layers:
            for name, p in gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(p[hidden:2 * hidden], 1.0)

    def _sweep(self, t: torch.Tensor):
        out = t
        last = len(self.layers) - 1
        for li, gru in enumerate(self.layers):
            out, _ = gru(out)
            # use residual connection
            if self.residual and out.shape == t.shape:
                out = out + t
            if li != last:
                out = self.inter_drop(out)
            t = out
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        # width sweep
        if self.axis == -1:
            t = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
            o = self._sweep(t)
            return o.view(B, H, W, -1).permute(0, 3, 1, 2)
        # height sweep
        else:
            t = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
            o = self._sweep(t)
            return o.view(B, W, H, -1).permute(0, 3, 2, 1)


class KspaceRNNEncoder(nn.Module):
    """
   Recurrent encoder for k-space features using bidirectional GRU sweeps.

   This module encodes k-space features by sweeping bidirectional GRUs
   along both horizontal and vertical axes, then fuses their outputs.
   The fusion can be either concatenation followed by a 1x1 convolution,
   or element-wise addition. In our experiment, we use concatenation.

   Parameters
   ----------
   in_ch : int
       Number of input channels.
   nf : int, default=64
       Hidden size of each GRU sweep.
   fuse : {"concat", "add"}, default="concat"
       Fusion method for horizontal and vertical sweeps:
       - "concat": concatenate and project with 1x1 convolution.
       - "add": element-wise addition.
   fuse_drop_p : float, default=0.0
       Dropout probability applied after fusion.

   Shape
   -------
   Input:  (B, C, H, W)
   Output: (B, out_ch, H, W)
   """
    def __init__(self, in_ch, nf=64, fuse="concat", fuse_drop_p: float = 0.0):
        super().__init__()
        self.fuse = fuse
        # two SweepGRUs, different direction
        self.horz = SweepGRU_2(in_ch, nf, axis=-1)
        self.vert = SweepGRU_2(in_ch, nf, axis=-2)
        # fuse methods
        if fuse == "add":
            self.post = nn.Identity()
            self.out_ch = nf * 2
        else:
            self.post = nn.Conv2d(nf * 4, nf * 2, 1)
            self.out_ch = nf * 2
        self.fuse_drop = nn.Dropout2d(fuse_drop_p) if fuse_drop_p > 0 else nn.Identity()

    def forward(self, x):
        h1, h2 = self.horz(x), self.vert(x)
        # fuse methods
        h = h1 + h2 if self.fuse == "add" else torch.cat([h1, h2], 1)
        return self.fuse_drop(self.post(h))


class CoarseHead(nn.Module):
    """
    Coarse prediction head for segmentation or reconstruction.

    This module applies a small convolutional block to reduce feature maps
    into a single-channel coarse output. It is typically used as an
    auxiliary head in multi-stage architectures.

    Parameters
    ----------
    in_ch : int
        Number of input feature channels.

    Shape
    -------
    Input:  (B, C, H, W)   where C = in_ch
    Output: (B, 1, H, W)   coarse prediction map
    """
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            # 2 convolution blocks
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(in_ch, 1, 1)
        )

    def forward(self, x): return self.net(x)


class ConvBlock(nn.Module):
    """
    Convolutional block with optional instance normalization and dropout.

    This block applies two sequential Conv2d layers with 3x3 kernels,
    each followed by normalization (instance norm or identity),
    ReLU activation, and optional spatial dropout.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    use_in : bool, default=True
        If True, applies InstanceNorm2d after each convolution.
        If False, skips normalization.
    drop_p : float, default=0.0
        Dropout probability for Dropout2d. If 0, dropout is skipped.

    Shape
    -----
    Input:  (B, in_ch, H, W)
    Output: (B, out_ch, H, W)
    """
    def __init__(self, in_ch, out_ch, use_in=True, drop_p: float = 0.0):
        super().__init__()
        norm = (lambda c: nn.InstanceNorm2d(c)) if use_in else (lambda c: nn.Identity())
        drop = (lambda: nn.Dropout2d(drop_p)) if drop_p > 0 else (lambda: nn.Identity())
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm(out_ch), nn.ReLU(True), drop(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm(out_ch), nn.ReLU(True), drop(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    U-Net upsampling block with skip connection.

    Parameters
    ----------
    in_ch : int
        Number of input channels (after concatenation with skip).
    out_ch : int
        Number of output channels after convolution.
    use_in : bool, default=True
        Whether to use instance normalization in ConvBlock.
    drop_p : float, default=0.0
        Dropout probability inside ConvBlock.

    Shape
    -----
    Input: (B, C_in, H_in, W_in)  # decoder feature
    Output: (B, out_ch, H_skip, W_skip)
    """
    def __init__(self, in_ch, out_ch, use_in=True, drop_p: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch, use_in, drop_p)

    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.size(-2) - x.size(-2), skip.size(-1) - x.size(-1)
        if dh or dw: x = F.pad(x, (0, dw, 0, dh))
        return self.conv(torch.cat([x, skip], 1))


class UNet(nn.Module):
    """
    U-Net for image segmentation.

    This model consists of an encoder-decoder structure with skip connections.
    The encoder path (downsampling) captures context, while the decoder path
    (upsampling) restores spatial resolution using skip connections from
    corresponding encoder layers.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int, default=1
        Number of output channels (e.g., segmentation classes or 1 for binary mask).
    base : int, default=24
        Number of base feature channels; doubled at each downsampling step.
    use_in : bool, default=True
        Whether to use InstanceNorm in convolutional blocks.
    num_pool : int, default=4
        Number of pooling (downsampling) stages in the encoder.

    Shape
    -----
    Input:  (B, in_ch, H, W)
    Output: (B, out_ch, H, W)
    """
    def __init__(self, in_ch, out_ch=1, base=24, use_in=True, num_pool=4):
        super().__init__()
        chs, self.pools, self.downs = [], nn.ModuleList(), nn.ModuleList()
        prev = in_ch
        for i in range(num_pool):
            # base
            c = base * 2 ** i
            # downstream
            self.downs.append(ConvBlock(prev, c, use_in))
            # max pooling
            self.pools.append(nn.MaxPool2d(2))
            chs.append(c)
            prev = c
        # bottleneck: conv block
        self.bottleneck = ConvBlock(prev, prev * 2, use_in)
        prev = prev * 2
        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(UpBlock(prev + c, c, use_in))
            prev = c
        # segmentation head
        self.final = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips, h = [], x
        for d, p in zip(self.downs, self.pools):
            h = d(h); skips.append(h); h = p(h)
        h = self.bottleneck(h)
        for up, sk in zip(self.ups, reversed(skips)):
            h = up(h, sk)
        return self.final(h)




# These implementation do not support dropout connections

# class SweepGRU_2_no_drop(nn.Module):
#     def __init__(self, in_ch, hidden, axis, num_layers: int = 3, residual: bool = True):
#         super().__init__()
#         assert axis in (-1, -2)
#         self.axis, self.residual = axis, residual
#
#         self.layers = nn.ModuleList([
#             nn.GRU(
#                 input_size=in_ch if i == 0 else hidden * 2,
#                 hidden_size=hidden,
#                 num_layers=1,
#                 batch_first=True,
#                 bidirectional=True
#             )
#             for i in range(num_layers)
#         ])
#
#         # forget‑gate bias init
#         for gru in self.layers:
#             for name, p in gru.named_parameters():
#                 if "bias" in name:
#                     nn.init.constant_(p[hidden:2 * hidden], 1.0)
#
#     def _sweep(self, t: torch.Tensor):
#         """
#         t: [BS, L, C] 序列。返回同形状输出（双向 × hidden）。
#         """
#         out = t
#         for gru in self.layers:
#             out, _ = gru(out)
#             if self.residual and out.shape == t.shape:
#                 out = out + t
#             t = out
#         return out
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         if self.axis == -1:  # width sweep
#             t = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
#             o = self._sweep(t)
#             return o.view(B, H, W, -1).permute(0, 3, 1, 2)
#         else:  # height sweep
#             t = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
#             o = self._sweep(t)
#             return o.view(B, W, H, -1).permute(0, 3, 2, 1)
#
#
#
# class KspaceRNNEncoder_no_drop(nn.Module):
#     def __init__(self, in_ch, nf=32, fuse="concat"):
#         super().__init__()
#         self.fuse = fuse
#         self.horz = SweepGRU_2(in_ch, nf, axis=-1)
#         self.vert = SweepGRU_2(in_ch, nf, axis=-2)
#         if fuse == "add":
#             self.post = nn.Identity()
#             self.out_ch = nf * 2
#         else:
#             self.post = nn.Conv2d(nf * 4, nf * 2, 1)
#             self.out_ch = nf * 2
#
#     def forward(self, x):
#         h1, h2 = self.horz(x), self.vert(x)
#         h = h1 + h2 if self.fuse == "add" else torch.cat([h1, h2], 1)
#         return self.post(h)
#
#
# class ConvBlock_no_drop(nn.Module):
#     def __init__(self, in_ch, out_ch, use_in=True):
#         super().__init__()
#         norm = (lambda c: nn.InstanceNorm2d(c)) if use_in else (lambda c: nn.Identity())
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1), norm(out_ch), nn.ReLU(True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1), norm(out_ch), nn.ReLU(True),
#         )
#
#     def forward(self, x): return self.block(x)
#
#
# class UpBlock_no_drop(nn.Module):
#     def __init__(self, in_ch, out_ch, use_in=True):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.conv = ConvBlock(in_ch, out_ch, use_in)
#
#     def forward(self, x, skip):
#         x = self.up(x)
#         dh, dw = skip.size(-2) - x.size(-2), skip.size(-1) - x.size(-1)
#         if dh or dw: x = F.pad(x, (0, dw, 0, dh))
#         return self.conv(torch.cat([x, skip], 1))
#
#
# class UNet_no_drop(nn.Module):
#     def __init__(self, in_ch, out_ch=1, base=32, use_in=True, num_pool=4):
#         super().__init__()
#         chs, self.pools, self.downs = [], nn.ModuleList(), nn.ModuleList()
#         prev = in_ch
#         for i in range(num_pool):
#             c = base * 2 ** i
#             self.downs.append(ConvBlock(prev, c, use_in))
#             self.pools.append(nn.MaxPool2d(2))
#             chs.append(c)
#             prev = c
#         self.bottleneck = ConvBlock(prev, prev * 2, use_in)
#         prev = prev * 2
#         self.ups = nn.ModuleList()
#         for c in reversed(chs):
#             self.ups.append(UpBlock(prev + c, c, use_in))
#             prev = c
#         self.final = nn.Conv2d(prev, out_ch, 1)
#
#     def forward(self, x):
#         skips, h = [], x
#         for d, p in zip(self.downs, self.pools):
#             h = d(h);
#             skips.append(h);
#             h = p(h)
#         h = self.bottleneck(h)
#         for up, sk in zip(self.ups, reversed(skips)):
#             h = up(h, sk)
#         return self.final(h)
