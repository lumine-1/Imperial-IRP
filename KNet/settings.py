"""
settings.py
===========

Centralized configuration file for the KNet project.

This file defines constants and default parameters used across
data preparation, training, and evaluation scripts. By storing
all settings here, experiments can be reproduced and modified
more easily without changing multiple files.

Contents
--------
1. **Data preparation settings**
   - Paths to raw cine MRI k-space data.
   - Output directories for reconstructed images (fully sampled and undersampled).

2. **Training settings**
   - Default training hyperparameters (epochs, batch size, learning rate, etc.).
   - Dataset split parameters (frames, cases range).
   - Default checkpoint filenames for different models.

3. **FastMRI settings**
   - Paths to the FastMRI dataset.
   - CSV file with labels or metadata.

Usage
-----
Import this file wherever configuration values are required.
For example::

    from settings import DEFAULTS, PATH_K

    print(DEFAULTS.epochs)        # 50
    print(PATH_K)                 # path to cine MRI ValidationSet

    # Override for custom experiments
    cfg = DEFAULTS
    cfg.epochs = 100

"""

from pathlib import Path
from types import SimpleNamespace

#################################################################### settings for prepare data

# Path to the raw cine MRI k-space data (Validation set).
PATH_K = "G:\CMRxRecon2023\ChallengeData_validation\ChallengeData_validation\SingleCoil\Cine\ValidationSet"

# Output directory for reconstructed fully sampled images
PATH_FULL_OUT = "prepared\\recon_full"

# Output directory for reconstructed undersampled images
PATH_UNDER_OUT = "prepared\\recon_undersampled"



#################################################################### settings for training

# Default training configuration, wrapped in a SimpleNamespace for convenience.
DEFAULTS = SimpleNamespace(
    data_root=Path("G:\CMRxRecon2023\ChallengeData_validation\ChallengeData_validation\SingleCoil\Cine\ValidationSet"),
    out_dir=Path("runs/checkpoints"),
    data_dir=Path("G:\CMRxRecon2023\pycharm"),
    epochs=50,
    batch=4,
    lr=4e-4,
    target_h=192,
    target_w=448,
    gpu=0,
    n_classes=4,
    num_workers=0,
    frames=tuple(range(12)),
    cases_from=1,
    cases_to=60,
    k_full_weight="final_k_full.pt",
    image_full_weight="final_image_full.pt",
    k_image_full_weight="final_k_image_full.pt",
    k_under_weight="final_k_under.pt",
    image_under_weight="final_image_under.pt",
    k_image_under_weight="final_k_image_under.pt",
    k_temp="temp.pt",
)



#################################################################### settings for fastmri

# Path to the raw FastMRI dataset (multi-coil version).
DATA_DIR_FAST = "G:\FastMRI\Dataset\multicoil_oringin"

# CSV file with labels or metadata for FastMRI dataset
TRAIN_CSV_FAST = "G:\FastMRI\Dataset\classify.csv"



