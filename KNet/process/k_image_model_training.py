"""
k_image_model_training.py

This script provides training routines for hybrid segmentation models
that combine k-space and image-domain features. It supports training
with both fully-sampled and undersampled reconstructions, enabling
multi-modal learning for improved segmentation.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from dataset import KImgSegDataset
from models.k_image_model import KImgSegModel2
from settings import DEFAULTS
from trainer import Trainer
from utils.main_utils import set_seed


def K_image_full_train(cfg: SimpleNamespace = DEFAULTS):
    """
    Train a segmentation model using both k-space and fully-sampled image features.
    This function constructs the joint dataset (k-space + image) from
    ``prepared`` and ``prepared/recon_full``, initializes the hybrid model,
    and trains it with the given configuration.

    Notes
    -----
    - Uses both k-space and image inputs for training.
    - Dataset is based on fully-sampled reconstructions.
    - Saves the best model based on validation mIoU.
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # related dataset
    train_ds = KImgSegDataset(
        k_root=Path("prepared"),
        img_root=Path("prepared\\recon_full"),
        split="train",
        undersample=False
    )
    val_ds = KImgSegDataset(
        k_root=Path("prepared"),
        img_root=Path("prepared\\recon_full"),
        split="val",
        undersample=False
    )

    train_loader = DataLoader(
        train_ds, cfg.batch, True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, cfg.batch, False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # load model
    model = KImgSegModel2()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_loader, val_loader, cfg.lr, cfg.out_dir, device, cfg.k_image_full_weight)

    for ep in range(1, cfg.epochs + 1):
        trainer.train_epoch(ep)
        trainer.validate(ep)

    print("K-image full Training done. Best mIoU:", trainer.best)
    print("")


def K_image_under_train(cfg: SimpleNamespace = DEFAULTS):
    """
    Train a segmentation model using both k-space and undersampled image features.
    This function constructs the joint dataset (k-space + image) from
    ``prepared`` and ``prepared/recon_undersampled``, applies undersampling,
    initializes the hybrid model, and trains it with the given configuration.

    Notes
    -----
    - Uses both k-space and image inputs for training.
    - Dataset is based on undersampled reconstructions with acceleration factor R.
    - Saves the best model based on validation mIoU.
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # related dataset
    train_ds = KImgSegDataset(
        k_root=Path("prepared"),
        img_root=Path("prepared\\recon_undersampled"),
        split="train",
        undersample=True,
        R_list=(24,), acs=2, seed=42
    )
    val_ds = KImgSegDataset(
        k_root=Path("prepared"),
        img_root=Path("prepared\\recon_undersampled"),
        split="val",
        undersample=True,
        R_list=(24,), acs=2, seed=42
    )

    train_loader = DataLoader(
        train_ds, cfg.batch, True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, cfg.batch, False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # load model
    model = KImgSegModel2()
    # model.load_state_dict(torch.load("runs/checkpoints/final_k_image_under.pt"))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_loader, val_loader, cfg.lr, cfg.out_dir, device, cfg.k_image_under_weight)

    for ep in range(1, cfg.epochs + 1):
        trainer.train_epoch(ep)
        trainer.validate(ep)

    print("K-image under Training done. Best mIoU:", trainer.best)
    print("")
