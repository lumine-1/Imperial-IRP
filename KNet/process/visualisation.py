"""
visualisation.py

This script provides visualisation utilities for evaluating trained
segmentation models on cine MRI datasets. It supports four modes:

Each function loads the corresponding validation dataset, restores
a pre-trained model from checkpoints, and visualizes predictions for
a selected sample.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from dataset import CineSegPreparedDataset, NpyImageSegDataset
from models.image_model import ImageSegModel
from models.k_model import KSegModel
from process.k_image_model_training import DEFAULTS
from utils.main_utils import set_seed
from utils.visualise_utils import visualize_single_sample, visualize_image_sample


class ModelCfg:
    in_ch: int = 2
    nf: int = 64
    num_classes: int = 4
    unet_base: int = 24
    use_in: bool = True


def K_full_vis(cfg: SimpleNamespace = DEFAULTS, index=12):
    """
   Visualize segmentation results on fully-sampled k-space input.
   This function loads the validation dataset without undersampling,
   restores a trained k-space segmentation model, and visualizes the
   prediction for a selected sample.

   Notes
   -----
   - Loads model weights from "runs/checkpoints/final_k_full.pt".
   - Uses CineSegPreparedDataset with `undersample=False`.
   """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    # related dataset
    val_ds = CineSegPreparedDataset(
        root_dir=Path("prepared/val"),
        target_shape=(cfg.target_h, cfg.target_w),
        undersample=False,
    )
    # load model
    model = KSegModel(ModelCfg()).to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_k_full.pt"))

    visualize_single_sample(model, val_ds, index=index, device=device)


def K_under_vis(cfg: SimpleNamespace = DEFAULTS, index=12):
    """
    Visualize segmentation results on undersampled k-space input.
    This function loads the validation dataset with undersampling
    (R=24, ACS=2), restores a trained k-space segmentation model,
    and visualizes the prediction for a selected sample.

    Notes
    -----
    - Loads model weights from "runs/checkpoints/final_k_under.pt".
    - Uses CineSegPreparedDataset with `undersample=True`.
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
    # load model
    model = KSegModel(ModelCfg()).to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_k_under.pt"))

    visualize_single_sample(model, val_ds, index=index, device=device)


def image_full_vis(cfg: SimpleNamespace = DEFAULTS, index=12):
    """
    Visualize segmentation results on fully-sampled image reconstructions.
    This function loads fully-sampled zero-filled images from the
    prepared dataset, restores a trained image segmentation model,
    and visualizes the prediction for a selected sample.

    Notes
    -----
    - Loads model weights from "runs/checkpoints/final_image_full.pt".
    - Uses NpyImageSegDataset with split="val".
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    root = Path("prepared/recon_full")
    # related dataset
    val_ds = NpyImageSegDataset(root, split="val", n_classes=4)
    # load model
    model = ImageSegModel().to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_image_full.pt"))
    visualize_image_sample(model, val_ds, index=index, device=device)


def image_under_vis(cfg: SimpleNamespace = DEFAULTS, index=12):
    """
    Visualize segmentation results on undersampled image reconstructions.
    This function loads undersampled zero-filled images from the prepared
    dataset, restores a trained image segmentation model, and visualizes
    the prediction for a selected sample.

    Notes
    -----
    - Loads model weights from "runs/checkpoints/final_image_under.pt".
    - Uses NpyImageSegDataset with split="val".
    """
    set_seed(42)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    root = Path("prepared/recon_undersampled")
    # related dataset
    val_ds = NpyImageSegDataset(root, split="val", n_classes=4)
    # load model
    model = ImageSegModel().to(device)
    model.load_state_dict(torch.load("runs/checkpoints/final_image_under.pt"))
    visualize_image_sample(model, val_ds, index=index, device=device)
