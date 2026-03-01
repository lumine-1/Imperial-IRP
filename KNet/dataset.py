"""
dataset.py

This module defines dataset classes for cine MRI segmentation and
reconstruction experiments. It provides loaders for raw k-space data,
preprocessed k-space arrays, reconstructed images, and joint k-space
+ image datasets.

Supported modes include:
- Directly reading k-space from .mat files with segmentation labels (.nii.gz).
- Loading preprocessed k-space slices stored as .npz files.
- Loading reconstructed image–mask pairs stored as .npy files.
- Providing aligned k-space and image pairs for joint training.

The datasets handle normalization, optional undersampling masks,
real/imaginary channel conversion, and alignment of segmentation
masks with k-space/image data. These classes are designed to be used
with PyTorch DataLoader for training and evaluating deep learning models.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re
import h5py
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.main_utils import resize_kspace_complex, center_crop_or_pad, make_cartesian_mask


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class CineSegDataset(Dataset):
    """
    Dataset for cine MRI segmentation directly from raw .mat k-space files and
    corresponding NIfTI segmentation labels.

    This dataset dynamically loads k-space data (from `.mat`) and segmentation
    labels (from `.nii.gz`) for each case and frame. It supports optional
    k-space undersampling with a Cartesian mask and outputs both under-sampled
    and full-sampled k-space in RI (real/imag) format.

    Returns
    -------
    k_us_ri : torch.FloatTensor
        Under-sampled k-space in RI format, shape (2*C, H, W).
    seg : torch.LongTensor
        Segmentation mask, shape (H, W).
    k_full_ri : torch.FloatTensor
        Full-sampled k-space in RI format, shape (2*C, H, W).
    """
    def __init__(self, root: Path, cases: List[str], frames: Tuple[int, ...],
                 target_shape: Tuple[int, int],
                 num_coil: int = 1, n_classes: int = 4,
                 undersample: bool = False, R_list: Tuple[int, ...] = (24,),
                 acs: int = 2, seed: int = 42):
        self.samples, self.tshape = [], target_shape
        self.num_coil, self.ncls = num_coil, n_classes
        self.undersample, self.R_list, self.acs = undersample, R_list, acs
        self.rng = np.random.RandomState(seed)

        for cid in cases:
            kf   = root / f"FullSample/{cid}/cine_sax.mat"
            labf = root / f"SegmentROI/{cid}/cine_sax_label.nii.gz"
            if not (kf.exists() and labf.exists()):
                continue
            with h5py.File(kf) as h:
                k_ds = h["kspace_single_full"]
                n_k  = k_ds.shape[1] if k_ds.ndim == 4 else k_ds.shape[0]
            n_lab = nib.load(str(labf)).shape[2]
            if n_k != n_lab:
                continue
            for f in frames:
                if f < n_k:
                    self.samples.append((kf, labf, f))
        print(f"Seg dataset size: {len(self.samples)} slices.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        kfile, labfile, f = self.samples[idx]

        # k-space
        with h5py.File(kfile, "r") as h:
            ds = h["kspace_single_full"]
            if ds.ndim == 3:  # [T,H,W]
                k = ds[f]  # -> [H,W]
                if getattr(k, "dtype", None) is not None and k.dtype.names:
                    k = k["real"] + 1j * k["imag"]
                k = k[None, ...]  # -> [1,H,W]
            else:  # [C,T,H,W]
                k = ds[:, f, :, :]  # -> [C,H,W]
                if getattr(k, "dtype", None) is not None and k.dtype.names:
                    k = k["real"] + 1j * k["imag"]
                k = k[: self.num_coil]

        # Center cut/zero filling to target shape
        k_rs = resize_kspace_complex(k, self.tshape)

        # under-sampling (optional)
        if self.undersample:
            H, W = self.tshape
            R = int(self.R_list[self.rng.randint(0, len(self.R_list))])
            mask = make_cartesian_mask(H, W, R=R, acs=self.acs, seed=int(self.rng.randint(1e9)))
            k_us = k_rs * mask[None, ...]
        else:
            k_us = k_rs

        # RI channel
        def to_ri(arr):
            return (np.stack([arr.real, arr.imag], axis=1)
                    .reshape(-1, *self.tshape)
                    .astype(np.float32))

        k_us_ri = torch.from_numpy(to_ri(k_us))
        k_full_ri = torch.from_numpy(to_ri(k_rs))

        # segmentation
        img = nib.load(str(labfile))
        seg = np.asarray(img.dataobj[:, :, f]).T.astype(np.int16)
        seg = center_crop_or_pad(seg, self.tshape)
        seg[seg >= self.ncls] = 0
        seg = torch.from_numpy(seg).long()

        return k_us_ri, seg, k_full_ri


class CineSegPreparedDataset(Dataset):
    """
    Dataset for cine MRI segmentation from preprocessed .npz files.

    This dataset loads k-space data and segmentation labels from `.npz` files
    prepared in advance. Each file stores real and imaginary parts of k-space,
    along with segmentation masks, aligned per frame. It supports optional
    k-space undersampling with a Cartesian mask.

    Returns
    -------
    k_us_ri : torch.FloatTensor
        Under-sampled k-space in RI format, shape (2*C, H, W).
    seg_t : torch.LongTensor
        Segmentation mask, shape (H, W).
    k_full_ri : torch.FloatTensor
        Full-sampled k-space in RI format, shape (2*C, H, W).

    Notes
    -------
    - This dataset assumes `.npz` files are pre-aligned and scaled.
    - Unlike `CineSegDataset`, no raw `.mat` or `.nii.gz` files are read at runtime.
    """
    def __init__(self,
                 root_dir: Path,
                 target_shape: Tuple[int, int],
                 undersample: bool = False,
                 R_list: Tuple[int, ...] = (24,),
                 acs: int = 2,
                 seed: int = 42):
        self.root_dir = Path(root_dir)
        self.tshape = target_shape
        self.undersample, self.R_list, self.acs = undersample, R_list, acs
        self.rng = np.random.RandomState(seed)

        self.files = sorted([p for p in self.root_dir.glob("*.npz")])
        assert len(self.files) > 0, f"No npz found in {self.root_dir}"
        self.index = []
        pat = re.compile(r"(P\d+)_f(\d+)\.npz$")
        for p in self.files:
            m = pat.search(p.name)
            if m:
                self.index.append((m.group(1), int(m.group(2))))
            else:
                self.index.append((p.stem, -1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as npz:
            k_real = npz["k_real"].astype(np.float32)
            k_imag = npz["k_imag"].astype(np.float32)
            seg = npz["seg"].astype(np.int16)

        # complex64 [C,H,W]
        k_rs = k_real.astype(np.float32) + 1j * k_imag.astype(np.float32)

        # regularization methods
        scale = np.percentile(np.abs(k_rs), 99.5) + 1e-8
        # scale = kscale_center_percentile(k_rs, center_ratio=0.25, p=99.5)

        # under-sampling (optional)
        if self.undersample:
            H, W = self.tshape
            R = int(self.R_list[self.rng.randint(0, len(self.R_list))])
            mask = make_cartesian_mask(H, W, R=R, acs=self.acs, seed=int(self.rng.randint(1e9)))
            k_us = k_rs * mask[None, ...]
        else:
            k_us = k_rs

        def to_ri(arr):
            return (np.stack([arr.real / scale, arr.imag / scale], axis=1)
                    .reshape(-1, *self.tshape).astype(np.float32))

        # use the same scale
        k_us_ri = torch.from_numpy(to_ri(k_us))
        k_full_ri = torch.from_numpy(to_ri(k_rs))
        seg_t = torch.from_numpy(seg).long()
        return k_us_ri, seg_t, k_full_ri


def _percentile_norm(x: np.ndarray, pmin=1.0, pmax=99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [pmin, pmax])
    y = (x - lo) / (hi - lo + 1e-8)
    return np.clip(y, 0, 1)


class NpyImageSegDataset(Dataset):
    """
    Dataset for loading image-mask pairs stored in NumPy (.npy) format.
    Each sample is a pair of an image (float32) and its corresponding
    segmentation mask (int16). Images are normalized by percentile scaling.

    Returns
    -------
    img : torch.FloatTensor
        Normalized image tensor of shape [1, H, W].
    msk : torch.LongTensor
        Segmentation mask tensor of shape [H, W], with values in [0, n_classes-1].
    """
    def __init__(self, root: Path, split: str = "train", n_classes: int = 4):
        super().__init__()
        # find the folder of image
        self.img_dir = root / split / "images"
        self.msk_dir = root / split / "masks"
        self.n_classes = n_classes

        self.items = []
        # stored in .npy
        for p in sorted(self.img_dir.rglob("*.npy")):
            base = p.stem
            if "_R" in base:
                base = base.split("_R")[0]
            case = p.parent.name
            m = self.msk_dir / case / f"{base}.npy"
            if m.exists():
                self.items.append((p, m))
        print(f"[NpyImageSegDataset:{split}] {len(self.items)} samples")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        # load image anf its mask
        p_img, p_msk = self.items[idx]
        img = np.load(p_img).astype(np.float32)
        msk = np.load(p_msk).astype(np.int16)

        img = _percentile_norm(img).astype(np.float32)
        img = torch.from_numpy(img)[None, ...]
        # load image anf its mask
        msk = torch.from_numpy(msk).long()
        msk[msk >= self.n_classes] = 0
        return img, msk


# Parse case ID and frame number from a filename stem.
def _parse_case_frame(stem: str) -> Tuple[str, int]:
    s = re.sub(r"_R\d+(_ACS\d+)?$", "", stem, flags=re.IGNORECASE)
    m = re.search(r"(P\d+)_f(\d+)", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse case/frame from: {stem}")
    case, f = m.group(1).upper(), int(m.group(2))
    return case, f


class KImgSegDataset(Dataset):
    """
    K-space + Image Segmentation Dataset.

    This dataset provides aligned pairs of k-space data, reconstructed
    images, and segmentation masks for training or evaluating
    joint reconstruction–segmentation models.

    It supports optional undersampling in k-space to simulate
    accelerated MRI acquisition.

    Returns
    -------
    x : torch.Tensor
        Input tensor, concatenating undersampled k-space (real/imag channels)
        and the image: shape [C_in, H, W].
    seg_t : torch.LongTensor
        Segmentation mask tensor: shape [H, W].
    k_full_ri : torch.Tensor
        Fully-sampled k-space (real/imag channels), normalized: shape [2*vc, H, W].

    Notes
    -----
    - Normalization: scaled by its 99.5th percentile (same to previous method) .
    - If undersampling is enabled, a random mask is applied to k-space.
    - The dataset matches (case, frame) between k-space and image files
      by parsing filenames, raising an error if no pairs are found.
    """
    def __init__(self,
                 k_root: Path,
                 img_root: Path,
                 split: str = "train",
                 target_shape: Tuple[int, int] = (192, 448),
                 n_classes: int = 4,
                 undersample: bool = False,
                 R_list: Tuple[int, ...] = (24,),
                 acs: int = 2,
                 seed: int = 42):
        super().__init__()
        # load data from disk
        self.k_dir = Path(k_root) / split
        self.img_dir = Path(img_root) / split / "images"
        self.tshape = target_shape
        self.n_classes = n_classes
        self.undersample, self.R_list, self.acs = undersample, R_list, acs
        self.rng = np.random.RandomState(seed)

        k_files = sorted(self.k_dir.glob("*.npz"))
        if not k_files:
            raise FileNotFoundError(f"No k npz under {self.k_dir.resolve()}")

        # build index: (case,frame) -> path
        img_index = {}
        for p in self.img_dir.rglob("*.npy"):
            try:
                key = _parse_case_frame(p.stem)
                img_index[key] = p
            except ValueError:
                continue

        pairs, miss_img = [], 0
        for p in k_files:
            # same reading methods
            try:
                key = _parse_case_frame(p.stem)
            except ValueError:
                continue
            q = img_index.get(key)
            if q is None:
                miss_img += 1
                continue
            pairs.append((p, q))

        self.pairs = pairs
        print(f"[KImgSegDataset:{split}] k={len(k_files)}, img_index={len(img_index)}, "
              f"matched={len(pairs)}, miss_img={miss_img}, root_k={self.k_dir}, root_img={self.img_dir}")

        if len(self.pairs) == 0:
            raise RuntimeError("No matched (k,img) pairs. Check naming or paths.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p_k, p_img = self.pairs[idx]

        # read k-space from npz
        with np.load(p_k, allow_pickle=False) as npz:
            k_real = npz["k_real"].astype(np.float32)
            k_imag = npz["k_imag"].astype(np.float32)
            seg = npz["seg"].astype(np.int16)

        k_rs = k_real + 1j * k_imag
        scale = np.percentile(np.abs(k_rs), 99.5) + 1e-8

        # under-sampling
        if self.undersample:
            H, W = self.tshape
            R = int(self.R_list[self.rng.randint(0, len(self.R_list))])
            mask = make_cartesian_mask(H, W, R=R, acs=self.acs, seed=int(self.rng.randint(1e9)))
            k_us = k_rs * mask[None, ...]
        else:
            k_us = k_rs

        def to_ri(arr):
            return (np.stack([arr.real / scale, arr.imag / scale], axis=1)
                    .reshape(-1, *self.tshape).astype(np.float32))

        # k space
        k_us_ri = torch.from_numpy(to_ri(k_us))
        k_full_ri = torch.from_numpy(to_ri(k_rs))

        # image
        img = np.load(p_img).astype(np.float32)
        img = _percentile_norm(img)
        img_t = torch.from_numpy(img)[None, ...]

        # segmentation
        seg_t = torch.from_numpy(seg).long()
        seg_t[seg_t >= self.n_classes] = 0
        x = torch.cat([k_us_ri, img_t], dim=0)

        return x, seg_t, k_full_ri

