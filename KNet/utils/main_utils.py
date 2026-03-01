from __future__ import annotations
"""
main_utils.py

Main utils for this project.
"""

from typing import List, Tuple
import torch.nn.functional as F
from pathlib import Path
import h5py, nibabel as nib, numpy as np, torch
import os, random
import numpy as np
import torch


def center_crop_or_pad(arr: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """
    Center crop or zero-pad a 2D array to the target shape.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array of shape (H, W).
    target : tuple of int
        Desired output size (th, tw).
    """
    H, W = arr.shape
    th, tw = target
    out = np.zeros(target, dtype=arr.dtype)
    h0s, w0s = max((H - th) // 2, 0), max((W - tw) // 2, 0)
    h0d, w0d = max((th - H) // 2, 0), max((tw - W) // 2, 0)
    hlen, wlen = min(th, H), min(tw, W)
    out[h0d:h0d + hlen, w0d:w0d + wlen] = arr[h0s:h0s + hlen, w0s:w0s + wlen]
    return out


def resize_kspace_complex(k: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """
    Resize complex k-space data to the target shape by center crop or zero-pad.

    Parameters
    ----------
    k : np.ndarray
        Input k-space array, shape (C, H, W) or (H, W).
    target : tuple of int
        Desired spatial size (th, tw).
    """
    if k.ndim == 2:
        k = k[None]
    C, H, W = k.shape
    th, tw = target
    out = np.zeros((C, th, tw), k.dtype)
    h0s, w0s = max((H - th) // 2, 0), max((W - tw) // 2, 0)
    h0d, w0d = max((th - H) // 2, 0), max((tw - W) // 2, 0)
    hlen, wlen = min(th, H), min(tw, W)
    out[:, h0d:h0d + hlen, w0d:w0d + wlen] = k[:, h0s:h0s + hlen, w0s:w0s + wlen]
    return out


def dice_loss(logits, target, eps=1e-5):
    """
    Compute soft Dice loss for segmentation.

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs of shape (B, C, H, W), before softmax.
    target : torch.Tensor
        Ground truth segmentation mask of shape (B, H, W), integer labels.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Scalar Dice loss (1 - mean Dice score).
    """
    probs = logits.softmax(1)  # [B, C, H, W]
    tgt1h = F.one_hot(target, logits.size(1)).movedim(-1, 1).float()  # [B, C, H, W]

    inter = (probs * tgt1h).flatten(2).sum(-1)  # [B, C]
    union = probs.flatten(2).sum(-1) + tgt1h.flatten(2).sum(-1)  # [B, C]

    dice = (2 * inter + eps) / (union + eps)  # [B, C]
    return 1 - dice.mean()


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # when use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_cartesian_mask(H, W, R=2, acs=24, seed=None, vd_power=2.0):
    """
    Generate a Cartesian undersampling mask along the phase-encoding (ky) direction.

    Parameters
    ----------
    H : int
        Number of k-space lines along ky (height).
    W : int
        Number of k-space columns along kx (width).
    R : int, default=2
        Acceleration factor (sampling rate ≈ 1/R).
    acs : int, default=24
        Number of fully-sampled central ACS lines.
    seed : int or None, default=None
        Random seed for reproducible sampling.
    vd_power : float, default=2.0
        Power for variable-density sampling (higher → more central lines).

    Returns
    -------
    mask : np.ndarray of shape (H, W), dtype=float32
        Binary sampling mask (1=keep, 0=discard).

    Notes
    -----
    - Always keeps the central `acs` lines.
    - Selects remaining lines with a probability biased toward
      the center of k-space (variable density).
    """
    rng = np.random.RandomState(seed)
    mask_ky = np.zeros(H, dtype=np.uint8)

    # ACS
    acs = int(min(max(acs, 0), H))
    if acs > 0:
        c0 = (H - acs) // 2
        mask_ky[c0:c0+acs] = 1

    # Randomly select some non-ACS lines to reach the target R
    target_keep = int(round(H / max(R, 1)))
    remain = max(target_keep - mask_ky.sum(), 0)
    if remain > 0:
        ky = np.linspace(-1, 1, H)
        w = (1.0 - np.abs(ky)) ** vd_power
        w[mask_ky == 1] = 0
        if w.sum() == 0:
            cand = np.where(mask_ky == 0)[0]
            choose = rng.choice(cand, size=min(remain, cand.size), replace=False)
        else:
            w = w / w.sum()
            choose = rng.choice(H, size=remain, replace=False, p=w)
        mask_ky[choose] = 1

    mask = np.repeat(mask_ky[:, None], W, axis=1)
    return mask.astype(np.float32)


def prepare_frames(
    data_root: Path,
    out_root: Path,
    cases: list[str],
    frames: tuple[int, ...] = tuple(range(12)),
    target_shape: tuple[int, int] = (192, 448),
    num_coil: int = 1,
    n_classes: int = 4,
    val_stride: int = 5,
    val_folder_name: str = "val",
):
    """
    Prepare and save k-space slices with segmentation labels.
    This function reads MRI data, extracts specified frames,
    crops/pads k-space to a fixed target shape, and pairs each slice
    with its segmentation mask. The processed data are saved as .npz files.

    Parameters
    ----------
    data_root : Path
        Root directory containing FullSample/<case>/cine_sax.mat and
        SegmentROI/<case>/cine_sax_label.nii.gz.
    out_root : Path
        Output directory where processed .npz files are saved.
    cases : list of str
        List of case IDs (e.g. ["P001", "P002", ...]).
    frames : tuple of int
        Frame indices to extract from each case.
    target_shape : tuple of int
        Target spatial resolution (H, W).
    num_coil : int
        Number of coils to keep.
    n_classes : int
        Number of segmentation classes. Labels >= n_classes are set to 0.
    val_stride : int
        Every `val_stride`-th case is assigned to validation set.
    val_folder_name : str
        Name of the validation output folder.

    Output
    ------
    - train/*.npz : training k-space slices
    - <val_folder_name>/*.npz : validation k-space slices
    Each saved .npz contains:
        k_real : float32 [C,H,W]
        k_imag : float32 [C,H,W]
        seg    : int16   [H,W]
        case   : str
        frame  : int
    """
    data_root = Path(data_root)
    out_root = Path(out_root)
    out_train = out_root / "train"
    out_val = out_root / val_folder_name
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # this dataset segment method is the same as others
    saved_train = saved_val = 0
    val_ids = set(cases[::5])
    # read k space and its mask
    for i, cid in enumerate(cases, 1):
        kf = data_root / f"FullSample/{cid}/cine_sax.mat"
        labf = data_root / f"SegmentROI/{cid}/cine_sax_label.nii.gz"
        if not kf.exists() or not labf.exists():
            print(f"[{cid}] skip: missing file kf={kf.exists()} labf={labf.exists()}")
            continue

        try:
            with h5py.File(kf, "r") as h:
                ds = h["kspace_single_full"]
                n_k  = ds.shape[1] if ds.ndim == 4 else ds.shape[0]
            n_lab = nib.load(str(labf)).shape[2]
        except Exception as e:
            print(f"[{cid}] skip: load error {e}")
            continue

        if n_k != n_lab:
            print(f"[{cid}] skip: frame mismatch k={n_k} lab={n_lab}")
            continue

        # dst_dir = out_val if (i % val_stride == 0) else out_train
        dst_dir = out_val if cid in val_ids else out_train

        # check frames
        for f in frames:
            if f >= n_k:
                print(f"[{cid}] skip frame f={f} (n_k={n_k})")
                continue

            with h5py.File(kf, "r") as h:
                ds = h["kspace_single_full"]
                if ds.ndim == 3:
                    k = ds[f]
                    if getattr(k, "dtype", None) is not None and getattr(k, "dtype").names:
                        k = k["real"] + 1j * k["imag"]
                    k = k[None, ...]
                else:
                    k = ds[:, f, :, :]
                    if getattr(k, "dtype", None) is not None and getattr(k, "dtype").names:
                        k = k["real"] + 1j * k["imag"]
                    k = k[: num_coil]

            k_rs = resize_kspace_complex(k, target_shape).astype(np.complex64)

            img = nib.load(str(labf))
            seg = np.asarray(img.dataobj[:, :, f]).T.astype(np.int16)
            seg = center_crop_or_pad(seg, target_shape)
            seg[seg >= n_classes] = 0

            # save files for subsequent usage
            out_path = dst_dir / f"{cid}_f{f:02d}.npz"
            np.savez_compressed(
                out_path,
                k_real=k_rs.real.astype(np.float32),
                k_imag=k_rs.imag.astype(np.float32),
                seg=seg.astype(np.int16),
                case=cid, frame=f,
            )
            if dst_dir is out_train: saved_train += 1
            else:                    saved_val   += 1
        print(f"[{cid}] done -> {dst_dir}")

    print(f"Saved: train={saved_train} files, {val_folder_name}={saved_val} files")


def load_partial_weights(model, ckpt_path, map_location="cpu",
                         adapt_first_conv=True, verbose=True):
    """
    Load a checkpoint into a model, only keeping matching weights.

    Returns
    -------
    msg : IncompatibleKeys
        PyTorch object containing missing and unexpected keys after loading.

    Notes
    -----
    - Unexpected keys (present in checkpoint but not in model) are ignored.
    - Mismatched keys (name matches but shape differs) are skipped,
      unless adapted explicitly.
    """
    # read checkpoint safe
    sd_raw = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd_raw = sd_raw["state_dict"]

    # remove 'module.'
    sd_raw = { (k[7:] if k.startswith("module.") else k): v for k, v in sd_raw.items() }
    model_sd = model.state_dict()
    new_sd = {}
    mismatched = []
    unexpected = []
    for k, v in sd_raw.items():
        if k not in model_sd:
            unexpected.append(k)
            continue
        if model_sd[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        new_sd[k] = v

    # special cases
    if adapt_first_conv:
        # real name of first layer
        kname = "refiner.downs.0.block.0.weight"
        if kname in sd_raw and kname in model_sd:
            w_old = sd_raw[kname]
            w_new = model_sd[kname]
            if w_old.ndim == 4 and w_new.ndim == 4 and w_old.shape[1] == 1 and w_new.shape[1] == 2 \
               and w_old.shape[0] == w_new.shape[0] and w_old.shape[2:] == w_new.shape[2:]:
                # replicate to two channels
                w_adapt = w_old.repeat(1, 2, 1, 1) / 2.0
                new_sd[kname] = w_adapt
                mismatched = [(k, a, b) for (k,a,b) in mismatched if k != kname]
                if verbose:
                    print(f"[adapt] {kname}: {tuple(w_old.shape)} -> {tuple(w_new.shape)} (replicate/avg)")

    # loading
    missing_before = [k for k in model_sd.keys() if k not in new_sd]
    msg = model.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"[load] loaded {len(new_sd)}/{len(model_sd)} params from ckpt.")
        if unexpected:
            print(f"[load] unexpected in ckpt: {len(unexpected)} (e.g. {unexpected[:3]})")
        if mismatched:
            print(f"[load] mismatched shapes: {len(mismatched)} (e.g. {mismatched[:3]})")
        # return IncompatibleKeys(missing/unused)
        if hasattr(msg, "missing_keys") and msg.missing_keys:
            print(f"[load] missing in model after filter: {len(msg.missing_keys)} (e.g. {msg.missing_keys[:5]})")

    return msg




