"""
k_model_training.py

This script provides training routines for segmentation models
that operate purely in the k-space domain. It sets up datasets,
builds models, and runs training/validation loops for both
fully-sampled and undersampled settings.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from dataset import CineSegDataset, CineSegPreparedDataset
from models.k_model import KSegModel, ModelCfg
from settings import DEFAULTS
from trainer import Trainer
from utils.main_utils import set_seed


def K_full_train(cfg: SimpleNamespace = DEFAULTS):
    """
    Train a segmentation model directly on fully-sampled k-space data.
    This function prepares the dataset from ``prepared/train`` and ``prepared/val``,
    builds the k-space based segmentation model, and trains it with the given configuration.

    Notes
    -----
    - Uses k-space as the primary input for segmentation.
    - Dataset is based on fully-sampled acquisitions.
    - Saves the best model checkpoint according to validation mIoU.
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # related dataset
    train_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/train"),
        target_shape=(cfg.target_h, cfg.target_w),
        undersample=False,
    )
    val_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/val"),
        target_shape=(cfg.target_h, cfg.target_w),
        undersample=False,
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
    model = KSegModel(ModelCfg(num_classes=cfg.n_classes))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_loader, val_loader, cfg.lr, cfg.out_dir, device, cfg.k_full_weight)

    for ep in range(1, cfg.epochs + 1):
        trainer.train_epoch(ep)
        trainer.validate(ep)

    print("K full Training done. Best mIoU:", trainer.best)
    print("")


def K_under_train(cfg: SimpleNamespace = DEFAULTS):
    """
    Train a segmentation model directly on undersampled k-space data.
    This function prepares the dataset from ``prepared/train`` and ``prepared/val``,
    applies undersampling with an acceleration factor, builds the k-space
    based segmentation model, and trains it with the given configuration.

    Notes
    -----
    - Uses undersampled k-space as the primary input for segmentation.
    - Undersampling factor is set via ``R_list`` and ``acs``.
    - Saves the best model checkpoint according to validation mIoU.
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # cases = [f"P{str(i).zfill(3)}" for i in range(cfg.cases_from, cfg.cases_to + 1)]
    # val_cases = cases[::5]
    # train_cases = [c for c in cases if c not in val_cases]
    # frames = tuple(cfg.frames)

    # related dataset
    train_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/train"),
        target_shape=(cfg.target_h, cfg.target_w),
        undersample=True,
        R_list=(24,), acs=2, seed=42
    )
    val_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/val"),
        target_shape=(cfg.target_h, cfg.target_w),
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
    model = KSegModel(ModelCfg(num_classes=cfg.n_classes))
    # model.load_state_dict(torch.load("runs/checkpoints/final_k_under.pt"))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_loader, val_loader, cfg.lr, cfg.out_dir, device, cfg.k_under_weight)

    for ep in range(1, cfg.epochs + 1):
        trainer.train_epoch(ep)
        trainer.validate(ep)

    print("K under Training done. Best mIoU:", trainer.best)
    print("")






