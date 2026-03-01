"""
recon_utils.py

Reconstruction utils for images.
"""

from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def make_case_split():
    """
    Split dataset cases into training and validation sets.
    All split methods are same in this project
    """
    cases = [f"P{str(i).zfill(3)}" for i in range(1, 60)]
    val_cases = cases[::5]
    train_cases = [c for c in cases if c not in val_cases]
    frames = tuple(range(12))
    return train_cases, val_cases, frames


def ifft2c_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2D inverse FFT with orthogonal normalization.
    """
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def _percentile_norm(x: np.ndarray, pmin=1.0, pmax=99.5) -> np.ndarray:
    """
    Normalize array values to [0,1] using percentile-based scaling.
    """
    lo, hi = np.percentile(x, [pmin, pmax])
    y = (x - lo) / (hi - lo + 1e-8)
    return np.clip(y, 0, 1)


# export. create new folder
def _ri_to_cplx_torch(k_ri: torch.Tensor, num_coil: int) -> torch.Tensor:
    """Convert RI-formatted k-space tensor into complex-valued representation."""
    C = num_coil
    H, W = k_ri.shape[-2], k_ri.shape[-1]
    k2 = k_ri.view(C, 2, H, W)
    return torch.complex(k2[:, 0], k2[:, 1])


def _zf_from_kri(k_ri: torch.Tensor, num_coil: int) -> np.ndarray:
    """Compute zero-filled reconstruction magnitude image (RSS for multi-coil)"""
    kc = _ri_to_cplx_torch(k_ri, num_coil=num_coil)  # [C,H,W] complex
    if kc.shape[0] == 1:
        img = ifft2c_torch(kc).abs().squeeze(0)
        return img.detach().cpu().numpy().astype(np.float32)
    else:
        img_c = ifft2c_torch(kc)
        mag = torch.abs(img_c)
        rss = torch.sqrt((mag ** 2).sum(dim=0))
        return rss.detach().cpu().numpy().astype(np.float32)


def export_undersampled_zf_images(
    ds: Dataset,
    out_dir: Path,
    split_name: str,
    num_coil: int = 1,
    save_png: bool = True,
):
    """
    Export zero-filled reconstructions and masks from a dataset, and record a manifest.

    result after export：
      - zf:  out_dir/split/images/<case>/{case}_f{frame}_R{R}_ACS{acs}.npy
      - seg: out_dir/split/masks/<case>/{case}_f{frame}.npy
      - manifest.csv record the path of idx/case/frame/H/W/R/ACS/.
    """
    out_img = out_dir / split_name / "images"
    out_msk = out_dir / split_name / "masks"
    out_png = out_dir / split_name / "preview"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)
    if save_png:
        out_png.mkdir(parents=True, exist_ok=True)

    R_disp = None
    try:
        if hasattr(ds, "R_list") and len(ds.R_list) == 1:
            R_disp = ds.R_list[0]
        ACS_disp = getattr(ds, "acs", None)
    except Exception:
        R_disp, ACS_disp = None, None

    manifest_path = out_dir / split_name / "manifest.csv"
    with open(manifest_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx", "case", "frame", "H", "W", "R", "ACS", "img_npy", "mask_npy"])

        for idx in tqdm(range(len(ds)), desc=f"Export {split_name}"):
            sample = ds[idx]  # (k_us_ri, seg, k_full_ri)
            if not (isinstance(sample, (list, tuple)) and len(sample) >= 2):
                raise ValueError("Dataset item must be (k_us_ri, seg, k_full_ri?)")

            k_us_ri, seg = sample[0], sample[1]
            # case / frame
            kf, labf, f = ds.samples[idx]
            case = Path(kf).parent.name
            frame = int(f)

            # ZF reconstruct
            zf = _zf_from_kri(k_us_ri, num_coil=num_coil)
            H, W = zf.shape

            # build folder based on cases
            img_dir_case = out_img / case
            msk_dir_case = out_msk / case
            img_dir_case.mkdir(parents=True, exist_ok=True)
            msk_dir_case.mkdir(parents=True, exist_ok=True)
            if save_png:
                (out_png / case).mkdir(parents=True, exist_ok=True)

            # file name
            tag = f"{case}_f{frame}"
            tag_full = f"{tag}_R{R_disp}_ACS{ACS_disp}" if (R_disp is not None and ACS_disp is not None) else tag

            img_path = img_dir_case / f"{tag_full}.npy"
            msk_path = msk_dir_case / f"{tag}.npy"
            np.save(img_path, zf)
            np.save(msk_path, seg.numpy().astype(np.int16))

            # save image
            if save_png:
                viz = _percentile_norm(zf)
                plt.figure(figsize=(4, 4))
                plt.imshow(viz, cmap="gray")
                plt.title(tag_full)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_png / case / f"{tag_full}.png", dpi=150)
                plt.close()

            writer.writerow([idx, case, frame, H, W, R_disp, ACS_disp, str(img_path), str(msk_path)])

    print(f"[Export] Done. Manifest: {manifest_path}")


def _cases_from_manifest(manifest_csv: Path) -> set[str]:
    """Read case IDs from a manifest file."""
    cases = set()
    if not manifest_csv.exists():
        return cases
    with open(manifest_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.add(row["case"])
    return cases


def assert_no_case_leakage(export_root: Path):
    """Check that training and validation sets do not share cases."""
    train_m = export_root / "train" / "manifest.csv"
    val_m   = export_root / "val"   / "manifest.csv"
    if not train_m.exists() or not val_m.exists():
        print("[Warn] manifest not found, skip leakage check.")
        return
    train_cases = _cases_from_manifest(train_m)
    val_cases   = _cases_from_manifest(val_m)
    overlap = train_cases & val_cases
    print(f"[Check] train cases: {len(train_cases)}, val cases: {len(val_cases)}, overlap: {len(overlap)}")
    if overlap:
        raise RuntimeError(f"Case leakage detected! Overlap: {sorted(list(overlap))[:10]} ...")
    print("[Check] No case leakage")