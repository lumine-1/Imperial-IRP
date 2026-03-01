"""
evaluation.py

This file provides evaluation routines for segmentation models
trained on different input modalities and sampling conditions.

It includes functions to evaluate:
- K-space only models (full-sampled and undersampled).
- Image-domain models using reconstructed images.
- Hybrid models combining k-space and image features.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from dataset import CineSegDataset, CineSegPreparedDataset, NpyImageSegDataset, KImgSegDataset
from evaluate import evaluate_segmentation
from models.image_model import ImageSegModel
from models.k_image_model import KImgSegModel2
from models.k_model import KSegModel
from settings import DEFAULTS
from utils.main_utils import set_seed


# model settings
class ModelCfg:
    in_ch: int = 2
    nf: int = 64
    num_classes: int = 4
    unet_base: int = 24
    use_in: bool = True


def K_full_evaluate(cfg: SimpleNamespace = DEFAULTS):
    """
    Evaluate segmentation performance using full-sampled k-space data.

    Parameters
    ----------
    cfg : SimpleNamespace, optional
        Configuration object with attributes such as `gpu`, `batch`,
        `target_h`, `target_w`, and `num_workers`.

    Notes
    -----
    - Uses CineSegPreparedDataset with fully sampled k-space.
    - Loads a model checkpoint trained on full-sampled k-space.
    - Reports mIoU, CE loss, Dice loss, and per-class metrics.
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    val_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/val"),
        target_shape=(cfg.target_h, cfg.target_w),
        undersample=False,
    )
    val_loader = DataLoader(
        val_ds, cfg.batch, False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # load model and weights
    model = KSegModel(ModelCfg()).to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_k_full.pt"))

    evaluate_segmentation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        voxel_spacing=(1.0, 1.0),
        print_per_class=True
    )


def K_under_evaluate(cfg: SimpleNamespace = DEFAULTS):
    """
    Evaluate segmentation performance using undersampled k-space data.

    Parameters
    ----------
    cfg : SimpleNamespace, optional
        Configuration object with attributes such as `gpu`, `batch`,
        `target_h`, `target_w`, `num_workers`, and sampling settings.

    Notes
    -----
    - Uses CineSegPreparedDataset with undersampling (R=24, ACS=2).
    - Loads a model checkpoint trained on undersampled k-space.
    - Reports segmentation metrics (mIoU, CE loss, Dice loss).
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # related dataset
    val_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/val"),
        target_shape=(cfg.target_h, cfg.target_w),
        undersample=True,
        R_list=(24,), acs=2, seed=42
    )
    val_loader = DataLoader(
        val_ds, cfg.batch, False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # load model and weights
    model = KSegModel(ModelCfg()).to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_k_under.pt"))

    evaluate_segmentation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        voxel_spacing=(1.0, 1.0),
        print_per_class=True
    )


def image_full_evaluate(cfg: SimpleNamespace = DEFAULTS):
    """
    Evaluate segmentation performance using reconstructed full-sampled images.

    Parameters
    ----------
    cfg : SimpleNamespace, optional
        Configuration object with GPU index and other evaluation parameters.

    Notes
    -----
    - Uses NpyImageSegDataset with images reconstructed from full k-space.
    - Loads a model checkpoint trained on reconstructed full-sampled images.
    - Computes and prints segmentation metrics with per-class results.
    """
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path("prepared/recon_full")
    batch = 4;

    # related dataset
    val_ds = NpyImageSegDataset(root, split="val", n_classes=4)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    # load model and weights
    model = ImageSegModel().to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_image_full.pt"))

    # set to (dy, dx) in mm if available
    evaluate_segmentation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        voxel_spacing=(1.0, 1.0),
        print_per_class=True
    )


def image_under_evaluate(cfg: SimpleNamespace = DEFAULTS):
    """
    Evaluate segmentation performance using reconstructed undersampled images.

    Parameters
    ----------
    cfg : SimpleNamespace, optional
        Configuration object with batch size, GPU index, etc.

    Notes
    -----
    - Uses NpyImageSegDataset with images reconstructed from undersampled k-space.
    - Loads a model checkpoint trained on reconstructed undersampled images.
    - Reports standard segmentation metrics (mIoU, CE loss, Dice loss).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path("prepared/recon_undersampled")

    # related dataset
    val_ds = NpyImageSegDataset(root, split="val", n_classes=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, pin_memory=True)

    # load model and weights
    model = ImageSegModel().to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_image_under.pt"))

    # set to (dy, dx) in mm if available
    evaluate_segmentation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        voxel_spacing=(1.0, 1.0),
        print_per_class=True
    )


def K_image_full_evaluate(cfg: SimpleNamespace = DEFAULTS):
    """
    Evaluate segmentation performance using both k-space and reconstructed
    full-sampled image features.

    Parameters
    ----------
    cfg : SimpleNamespace, optional
        Configuration object with `gpu`, `batch`, and dataloader settings.

    Notes
    -----
    - Uses KImgSegDataset combining k-space and reconstructed full-sampled images.
    - Loads a hybrid model checkpoint (KImgSegModel2).
    - Reports segmentation metrics including per-class scores.
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # related dataset
    val_ds = KImgSegDataset(
        k_root=Path("prepared"),
        img_root=Path("prepared\\recon_full"),
        split="val",
        undersample=False
    )
    val_loader = DataLoader(
        val_ds, cfg.batch, False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # load model and weights
    model = KImgSegModel2().to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_k_image_full.pt"))
    evaluate_segmentation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        voxel_spacing=(1.0, 1.0),
        print_per_class=True
    )


def K_image_under_evaluate(cfg: SimpleNamespace = DEFAULTS):
    """
   Evaluate segmentation performance using both k-space and reconstructed
   undersampled image features.

   Parameters
   ----------
   cfg : SimpleNamespace, optional
       Configuration object with GPU index, batch size, and sampling settings.

   Notes
   -----
   - Uses KImgSegDataset combining k-space and reconstructed undersampled images.
   - Loads a hybrid model checkpoint trained with undersampling (R=24, ACS=2).
   - Reports segmentation metrics including per-class results.
   """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    # related dataset
    val_ds = KImgSegDataset(
        k_root=Path("prepared"),
        img_root=Path("prepared\\recon_undersampled"),
        split="val",
        undersample=True,
        R_list=(24,), acs=2, seed=42
    )
    val_loader = DataLoader(
        val_ds, cfg.batch, False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # load model and weights
    model = KImgSegModel2().to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_k_image_under.pt"))
    evaluate_segmentation(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        voxel_spacing=(1.0, 1.0),
        print_per_class=True
    )
