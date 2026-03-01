"""
fastmri.py

End-to-end k-space reconstruction pretraining on a fastMRI-style dataset.
This script loads slice metadata from CSV, reads multicoil k-space from HDF5,
applies PCA coil compression and optional k-space augmentations (global phase,
linear phase ramp, flips), and trains a lightweight reconstruction network
(KspaceRNNEncoder + small CNN head) to predict RSS magnitude images.

Use this as a quick sanity check of model capacity on a different dataset
before running the downstream cine segmentation experiments. It handles data
prep, training loop (with AMP and grad-clipping), and checkpoint saving in
a single, self-contained file.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math, numpy as np, pandas as pd, h5py, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Tuple
from models.modules import KspaceRNNEncoder
from settings import DATA_DIR_FAST, TRAIN_CSV_FAST


# utils
def center_crop_or_pad_k(k: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    """Center-crop or zero-pad complex k-space data to a fixed target size."""
    C,H,W = k.shape; th,tw = target_hw
    out = np.zeros((C,th,tw), dtype=k.dtype)
    hs,ws = max((H-th)//2,0), max((W-tw)//2,0)
    hd,wd = max((th-H)//2,0), max((tw-W)//2,0)
    hlen,wlen = min(th,H), min(tw,W)
    out[:, hd:hd+hlen, wd:wd+wlen] = k[:, hs:hs+hlen, ws:ws+wlen]
    return out


def make_cartesian_mask(H: int, W: int, R: int, acs: int = 0, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate a Cartesian undersampling mask along the phase-encoding direction."""
    rng = rng or np.random.RandomState(None)
    mask = np.zeros((H,W), dtype=bool)
    if acs > 0:
        c0 = H//2 - acs//2
        mask[c0:c0+acs, :] = True
    if R > 1:
        offset = rng.randint(0, R)
        rows = np.arange(H)
        take = (rows % R) == offset
        if acs > 0: take[c0:c0+acs] = True
        mask[take, :] = True
    else:
        mask[:] = True
    return mask


def norm01(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Normalize tensor values to the [0,1] range."""
    x_min = x.amin(dim=(-2,-1), keepdim=True)
    x_max = x.amax(dim=(-2,-1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


# -------- dataset (recon pretrain) --------
class EnhancedKspaceSliceReconDataset(Dataset):
    """
    Dataset for k-space based image reconstruction pretraining.

    - Loads k-space slices from .h5 files.
    - Applies PCA coil compression, cropping, and optional data augmentations
      (global phase shift, linear phase ramp, horizontal/vertical flips).
    - For training mode, also applies under-sampling and complex Gaussian noise (augmentation).
    - Generates input k-space (real+imag channels) and ground truth image
      reconstructed from fully-sampled augmented k-space (RSS magnitude).

    Returns
    -------
    x : torch.FloatTensor
        Input k-space with real/imag channels, shape [2*vc, H, W].
    y : torch.FloatTensor
        Target magnitude image, shape [1, H, W].
    """
    def __init__(self, rows: pd.DataFrame, data_dir: str | Path,
                 max_virtual_coils: int = 8, target_hw: Tuple[int,int]=(320,320),
                 undersample_R: int = 1, acs: int = 0, noise_std: float = 1e-2,
                 train: bool = True, p_phase: float = 0.3, phase_coilwise: bool = False,
                 p_ramp: float = 0.3, max_shift_px: float = 8.0,
                 p_flipx: float = 0.2, p_flipy: float = 0.2, seed: int = 42):
        self.rows = rows.reset_index(drop=True)
        self.dir = Path(data_dir)
        self.vc = max_virtual_coils
        self.hw = target_hw
        self.R, self.acs = undersample_R, acs
        self.noise_std, self.train = noise_std, train
        self.p_phase, self.phase_coilwise = p_phase, phase_coilwise
        self.p_ramp, self.max_shift_px = p_ramp, max_shift_px
        self.p_fx, self.p_fy = p_flipx, p_flipy
        self.rng = np.random.RandomState(seed)

    def __len__(self): return len(self.rows)

    @staticmethod
    def _pca(k: np.ndarray, vc: int) -> np.ndarray:
        C,H,W = k.shape
        if C > vc:
            u,_,_ = np.linalg.svd(k.reshape(C,-1), full_matrices=False)
            k = (u[:,:vc].T @ k.reshape(C,-1)).reshape(vc,H,W)
        elif C < vc:
            k = np.concatenate([np.zeros((vc-C,H,W), k.dtype), k], 0)
        return k.astype(np.complex64)

    def _phase(self, k: np.ndarray) -> np.ndarray:
        c,_,_ = k.shape
        if self.phase_coilwise:
            phi = self.rng.uniform(0, 2*np.pi, size=(c,1,1)).astype(np.float32)
        else:
            phi = float(self.rng.uniform(0, 2*np.pi))
        return k * np.exp(1j * phi)

    def _ramp(self, k: np.ndarray) -> np.ndarray:
        c,H,W = k.shape
        kx, ky = np.fft.fftfreq(W), np.fft.fftfreq(H)
        KX,KY = np.meshgrid(kx, ky, indexing="xy")
        dx = self.rng.uniform(-self.max_shift_px, self.max_shift_px, size=(c,1,1))
        dy = self.rng.uniform(-self.max_shift_px, self.max_shift_px, size=(c,1,1))
        ramp = np.exp(1j * 2*np.pi * (dx*KX[None] + dy*KY[None])).astype(np.complex64)
        return k * ramp

    def __getitem__(self, i: int):
        fname, sli = self.rows.loc[i, "file"], int(self.rows.loc[i, "slice"])
        with h5py.File(self.dir / f"{fname}.h5", "r") as f:
            k = f["kspace"][sli]
        if k.ndim == 4: k = k[...,0] + 1j*k[...,1]
        k = self._pca(k, self.vc)
        k = center_crop_or_pad_k(k, self.hw)
        k_full = k.copy(); k_in = k.copy()

        # data augmentation
        if self.train and self.rng.rand() < self.p_phase:
            k_full = self._phase(k_full); k_in = self._phase(k_in)
        if self.train and self.rng.rand() < self.p_ramp:
            k_full = self._ramp(k_full);  k_in = self._ramp(k_in)
        if self.train:
            do_fx = self.rng.rand() < self.p_fx
            do_fy = self.rng.rand() < self.p_fy
            if do_fx: k_full = k_full[:,:,::-1]; k_in = k_in[:,:,::-1]
            if do_fy: k_full = k_full[:,::-1,:]; k_in = k_in[:,::-1,:]

        if self.train and self.R > 1:
            H,W = self.hw
            mask = make_cartesian_mask(H,W,self.R,self.acs,self.rng)
            k_in = k_in * mask[None]

        if self.train and self.noise_std > 0:
            n = (np.random.normal(0,self.noise_std,k_in.shape) +
                 1j*np.random.normal(0,self.noise_std,k_in.shape)).astype(np.complex64)
            k_in = (k_in + n).astype(np.complex64)

        # regularization
        scale = float(np.quantile(np.abs(k_full), 0.995)) + 1e-6
        k_full /= scale; k_in /= scale

        # label: k space convert to image
        img_cpx = np.fft.ifft2(k_full, norm="ortho", axes=(-2,-1))
        img_mag = np.sqrt((img_cpx.real**2 + img_cpx.imag**2).sum(0, dtype=np.float32))
        y = torch.from_numpy(img_mag[None].astype(np.float32))

        x = np.stack([k_in.real, k_in.imag], 1).reshape(-1, *k_in.shape[1:])
        x = torch.from_numpy(x.astype(np.float32))
        return x, y

# -------- model --------
class KspaceReconNet(nn.Module):
    """
    Simple reconstruction network for k-space pretraining.
    - Encodes k-space features with a recurrent encoder (KspaceRNNEncoder).
    - Decodes to a single-channel magnitude image via small CNN head.
    - Uses L1 loss between normalized prediction and target image.

    Methods
    -------
    forward(x):
        Forward pass, returns reconstructed image [B,1,H,W].
    recon_loss(pred, target_img):
        Computes normalized L1 reconstruction loss.
    """
    def __init__(self, in_ch: int, nf: int = 64, fuse: str = "concat"):
        super().__init__()
        self.enc = KspaceRNNEncoder(in_ch, nf, fuse)
        self.head = nn.Sequential(
            nn.Conv2d(self.enc.out_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x): return self.head(self.enc(x))

    @staticmethod
    def recon_loss(pred: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(norm01(pred), norm01(target_img))


# data split
def load_train_frames(csv_path: str) -> pd.DataFrame:
    """Load training frame metadata from a CSV file."""
    df = pd.read_csv(csv_path)
    assert {"file","slice"} <= set(df.columns)
    return df.reset_index(drop=True)

# -------- config --------
@dataclass
class CFG:
    DATA_DIR: str = DATA_DIR_FAST
    TRAIN_CSV: str = TRAIN_CSV_FAST
    MAX_VC: int = 4
    TARGET_HW: tuple[int,int] = (320,320)
    P_PHASE: float = 0.3
    NOISE_STD: float = 1e-2
    BATCH_SIZE: int = 4
    EPOCHS: int = 20
    LR: float = 2e-4
    NUM_WORKER: int = 0
    OUT_DIR: str = "runs/recon"
    SEED: int = 42
    FUSE: str = "concat"
    NF: int = 64
    CLIP_NORM: float = 1.0
    AMP: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# -------- train --------
def train_one_epoch(model, loader, opt, scaler, device):
    """
    Run one training epoch for the reconstruction model.

    Parameters
    ----------
    model : nn.Module
        The reconstruction network.
    loader : DataLoader
        Training data loader.
    opt : torch.optim.Optimizer
        Optimizer.
    scaler : torch.amp.GradScaler
        Mixed precision gradient scaler.
    device : torch.device
        Training device.
    """
    model.train(); running = 0.0; n = 0
    use_cuda_amp = (device.type == "cuda") and CFG.AMP
    pbar = tqdm(loader, desc="train", leave=False)
    # only use training set
    for x, img_gt in pbar:
        x = x.to(device, non_blocking=True)
        img_gt = img_gt.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_cuda_amp):
            pred = model(x)
            loss = model.recon_loss(pred, img_gt)
        scaler.scale(loss).backward()
        if CFG.CLIP_NORM is not None:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_NORM)
        scaler.step(opt); scaler.update()
        bs = x.size(0); running += loss.item()*bs; n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(n,1)

def main():
    """
    Main training loop for k-space reconstruction pretraining.

    - Loads training dataset from CSV.
    - Builds DataLoader and model.
    - Trains for the configured number of epochs.
    - Saves the trained model checkpoint and training loss history.
    """
    torch.manual_seed(CFG.SEED); np.random.seed(CFG.SEED)
    device = torch.device(CFG.DEVICE)
    out_dir = Path(CFG.OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = load_train_frames(CFG.TRAIN_CSV)

    # create dataset
    ds = EnhancedKspaceSliceReconDataset(
        df, CFG.DATA_DIR, max_virtual_coils=CFG.MAX_VC, target_hw=CFG.TARGET_HW,
        undersample_R=1, acs=0, noise_std=CFG.NOISE_STD, train=True, p_phase=CFG.P_PHASE
    )
    dl = DataLoader(ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                    num_workers=CFG.NUM_WORKER, pin_memory=True,
                    persistent_workers=CFG.NUM_WORKER>0)

    # load network, optimiser, scaler
    model = KspaceReconNet(in_ch=CFG.MAX_VC*2, nf=CFG.NF, fuse=CFG.FUSE).to(device)
    opt = AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    scaler = GradScaler(device.type, enabled=(device.type=="cuda" and CFG.AMP))

    losses = []
    for ep in range(1, CFG.EPOCHS+1):
        tr = train_one_epoch(model, dl, opt, scaler, device)
        losses.append(tr)
        print(f"[Epoch {ep:03d}] train={tr:.4f}")

    # save model weights
    torch.save({"model": model.state_dict(), "cfg": CFG.__dict__, "loss": losses},
               out_dir / "recon_model.pt")
    print(f"Done. Last train loss={losses[-1]:.4f}. Saved to {out_dir/'recon_model.pt'}")

if __name__ == "__main__":
    main()
