from pathlib import Path
import pytest
import torch

from dataset import CineSegDataset, NpyImageSegDataset
from settings import PATH_K


def _exists_nonempty_dir(p: Path) -> bool:
    return p.exists() and any(p.iterdir())

def test_npy_image_dataset_load():
    root = Path("../prepared/recon_full")
    if not _exists_nonempty_dir(root):
        pytest.skip("prepared/recon_full not found; skipping NpyImageSegDataset test")

    ds = NpyImageSegDataset(root, split="train", n_classes=4)
    if len(ds) == 0:
        pytest.skip("prepared/recon_full has no samples; skipping")

    img, msk = ds[0]
    assert isinstance(img, torch.Tensor) and img.ndim == 3 and img.shape[0] == 1
    assert isinstance(msk, torch.Tensor) and msk.ndim == 2

def test_cinesegdataset_smoke():
    raw_root = Path(PATH_K)
    if not raw_root.exists():
        pytest.skip("raw cine root not found; skipping CineSegDataset test")

    ds = CineSegDataset(
        root=raw_root,
        cases=["P001"],
        frames=(0,),
        target_shape=(192, 448),
        num_coil=1, n_classes=4,
        undersample=False
    )
    if len(ds) == 0:
        pytest.skip("CineSegDataset yielded 0 samples for given case/frame; skipping")

    k_ri, seg, k_full = ds[0]
    assert isinstance(k_ri, torch.Tensor) and k_ri.ndim == 3
    assert isinstance(seg, torch.Tensor) and seg.ndim == 2
    assert k_full.shape == k_ri.shape
