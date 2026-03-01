"""
prepare_data.py

This script provides utilities to preprocess cine MRI datasets for
segmentation and reconstruction experiments. It generates both k-space
and image-domain data in standardized formats, ready for training.

Output
------
Creates structured folders under ``prepared/`` containing k-space,
fully-sampled reconstructions, and undersampled reconstructions,
each split into training and validation sets.
"""

from pathlib import Path

from dataset import CineSegDataset
from utils.recon_utils import export_undersampled_zf_images, assert_no_case_leakage, make_case_split
from utils.main_utils import set_seed, prepare_frames


def prepare_k_space(path_k):
    """
    Prepare k-space data for segmentation/reconstruction experiments.
    This function reads cine MRI validation cases, extracts specified frames,
    and saves preprocessed k-space arrays into an output directory.

    Output
    ------
    A folder structure containing preprocessed k-space data, split into
    training and validation sets.
    """
    set_seed(42)
    data_root = Path(path_k)
    out_root = Path("prepared")
    cases = [f"P{str(i).zfill(3)}" for i in range(1, 60)]
    prepare_frames(
        data_root, out_root, cases,
        # keep val_cases = cases[::5] !
        val_stride=5,
        frames=tuple(range(12)),
        target_shape=(192, 448),
        num_coil=1, n_classes=4
    )



def prepare_image_full(path_k, path_full_out):
    """
    Prepare fully-sampled images for reconstruction training.
    This function loads cine MRI datasets, splits cases into training and
    validation sets, and exports corresponding zero-filled (ZF) images
    without undersampling.

    Output
    ------
    A folder structure under "prepared/recon_full" containing fully-sampled
    zero-filled images for training and validation.
    """
    # fixed R/ACS
    target_shape = (192, 448)
    data_root = Path(path_k)
    out_dir = Path(path_full_out)
    train_cases, val_cases, frames = make_case_split()

    # export training set
    train_ds_us = CineSegDataset(
        data_root, train_cases, frames, target_shape,
        num_coil=1, n_classes=4,
        undersample=False
    )
    export_undersampled_zf_images(train_ds_us, out_dir, split_name="train", num_coil=1, save_png=True)

    # export validation set
    val_ds_us = CineSegDataset(
        data_root, val_cases, frames, target_shape,
        num_coil=1, n_classes=4,
        undersample=False
    )
    export_undersampled_zf_images(val_ds_us, out_dir, split_name="val", num_coil=1, save_png=True)

    # leakage checking after exporting
    assert_no_case_leakage(out_dir)



def prepare_image_under(path_k, path_under_out):
    """
    Prepare undersampled images for reconstruction training.
    This function loads cine MRI datasets, applies undersampling with a
    specified acceleration factor (R) and ACS lines, and exports
    corresponding zero-filled (ZF) reconstructions.

    Output
    ------
    A folder structure under "prepared/recon_undersampled" containing
    undersampled zero-filled images for training and validation.
    """
    # fixed R/ACS
    target_shape = (192, 448)
    R, ACS = 24, 2
    data_root = Path(path_k)
    out_dir = Path(path_under_out)

    train_cases, val_cases, frames = make_case_split()

    # export training set
    train_ds_us = CineSegDataset(
        data_root, train_cases, frames, target_shape,
        num_coil=1, n_classes=4,
        undersample=True, R_list=(R,), acs=ACS, seed=42
    )
    export_undersampled_zf_images(train_ds_us, out_dir, split_name="train", num_coil=1, save_png=True)

    # export validation set
    val_ds_us = CineSegDataset(
        data_root, val_cases, frames, target_shape,
        num_coil=1, n_classes=4,
        undersample=True, R_list=(R,), acs=ACS, seed=42
    )
    export_undersampled_zf_images(val_ds_us, out_dir, split_name="val", num_coil=1, save_png=True)

    # leakage checking after exporting
    assert_no_case_leakage(out_dir)








