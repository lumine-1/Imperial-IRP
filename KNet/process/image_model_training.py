"""
image_model_training.py

This script provides training routines for image-domain segmentation
models based on reconstructed MRI images. It loads prepared datasets,
constructs U-Net models, and trains them on both fully-sampled and
undersampled reconstructions.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from dataset import NpyImageSegDataset
from models.image_model import ImageSegModel
from settings import DEFAULTS
from trainer import Trainer
from utils.main_utils import set_seed


def image_full_train(cfg: SimpleNamespace = DEFAULTS):
    """
    Train an image-only segmentation model using fully-sampled reconstructions.
    This function loads the prepared dataset from ``prepared/recon_full``,
    builds the UNet-based segmentation model, and trains it with the
    specified configuration.

    Notes
    -----
    - Uses cross-entropy and dice loss for optimization.
    - Saves the best model based on validation mIoU.
    - Batch size is fixed at 4 for full-sampled training.
    """
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path("prepared/recon_full")
    batch = 4;

    # related dataset
    train_ds = NpyImageSegDataset(root, split="train", n_classes=4)
    val_ds = NpyImageSegDataset(root, split="val", n_classes=4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    # load model
    model = ImageSegModel()
    trainer = Trainer(model, train_loader, val_loader, cfg.lr, cfg.out_dir, device, cfg.image_full_weight)

    for ep in range(1, cfg.epochs + 1):
        trainer.train_epoch(ep)
        trainer.validate(ep)

    print("Image full Training done. Best mIoU:", trainer.best)
    print("")


def image_under_train(cfg: SimpleNamespace = DEFAULTS):
    """
    Train an image-only segmentation model using undersampled reconstructions.
    This function loads the prepared dataset from ``prepared/recon_undersampled``,
    builds the UNet-based segmentation model, and trains it with the
    specified configuration.

    Notes
    -----
    - Uses cross-entropy and dice loss for optimization.
    - Saves the best model based on validation mIoU.
    - Batch size is configurable via ``cfg.batch``.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path("prepared/recon_undersampled")

    # related dataset
    train_ds = NpyImageSegDataset(root, split="train", n_classes=4)
    val_ds = NpyImageSegDataset(root, split="val", n_classes=4)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, pin_memory=True)

    # load model
    model = ImageSegModel()
    trainer = Trainer(model, train_loader, val_loader, cfg.lr, cfg.out_dir, device, cfg.image_under_weight)

    for ep in range(1, cfg.epochs + 1):
        trainer.train_epoch(ep)
        trainer.validate(ep)

    print("Image under Training done. Best mIoU:", trainer.best)
    print("")
