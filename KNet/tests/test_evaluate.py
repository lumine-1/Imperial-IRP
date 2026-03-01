import pytest
import torch
from process import evaluation
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from evaluate import evaluate_segmentation, metric_hard_dice


class TwoTupleDataset(Dataset):
    def __init__(self, segs):
        self.segs = segs

    def __len__(self):
        return len(self.segs)

    def __getitem__(self, i):
        seg = self.segs[i]
        # encode seg into a channel so the model can recover it
        x = seg.float().unsqueeze(0)
        return x, seg


class ThreeTupleDataset(Dataset):
    def __init__(self, segs):
        self.segs = segs

    def __len__(self):
        return len(self.segs)

    def __getitem__(self, i):
        seg = self.segs[i]
        x = seg.float().unsqueeze(0)
        # any tensor with same H,W is fine
        k_full = x.clone()
        return x, seg, k_full


class PerfectCopyModel(nn.Module):
    def forward(self, x):
        mask = (x[:, :1, ...] > 0).float()
        scale = 50.0  # makes CE ~ 0
        logits = torch.cat([scale * (1.0 - mask), scale * mask], 1)
        return logits


def test_evaluate_functions_exist():
    # functions exists
    assert hasattr(evaluation, "K_full_evaluate")
    assert hasattr(evaluation, "image_full_evaluate")
    assert hasattr(evaluation, "K_image_full_evaluate")


def test_metric_hard_dice_perfect_binary():
    """metric_hard_dice should be 1.0 for perfect predictions (binary)."""
    B, H, W = 1, 4, 4
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, :2, :] = 1  # half ones, half zeros -> both classes present

    # Build perfect logits: class-1 where target==1 else class-0
    c1 = (target == 1).float().unsqueeze(1)
    c0 = 1.0 - c1
    logits = torch.cat([c0, c1], dim=1)

    score = metric_hard_dice(logits, target, exclude_bg=False)
    assert abs(score - 1.0) < 1e-6


def test_evaluate_segmentation_two_tuple_loader_perfect():
    device = torch.device("cpu")
    # build tiny batch of binary labels with both classes present
    segs = [
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
    ]
    ds = TwoTupleDataset(segs)
    dl = DataLoader(ds, batch_size=2, shuffle=False)

    model = PerfectCopyModel().to(device)
    metrics = evaluate_segmentation(
        model=model,
        val_loader=dl,
        device=device,
        num_classes=2,
        voxel_spacing=(1.0, 1.0),
        print_per_class=False,
    )

    assert set(metrics.keys()) == {
        "pixel_accuracy", "cross_entropy", "mean_dice_hard", "miou_metric", "overall_asd"
    }
    # perfect predictions
    assert abs(metrics["pixel_accuracy"] - 1.0) < 1e-6
    assert abs(metrics["mean_dice_hard"] - 1.0) < 1e-6
    assert abs(metrics["miou_metric"] - 1.0) < 1e-6
    # CE should be near zero (exact 0 given our logits construction)
    assert metrics["cross_entropy"] <= 0.5


def test_evaluate_segmentation_three_tuple_loader_perfect():
    device = torch.device("cpu")
    segs = [
        torch.tensor([[0, 1, 1],
                      [0, 0, 1]], dtype=torch.long),
        torch.tensor([[1, 0, 0],
                      [1, 1, 0]], dtype=torch.long),
    ]
    ds = ThreeTupleDataset(segs)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = PerfectCopyModel().to(device)
    metrics = evaluate_segmentation(
        model=model,
        val_loader=dl,
        device=device,
        num_classes=2,
        voxel_spacing=(1.0, 1.0),
        print_per_class=False,
    )

    assert metrics["pixel_accuracy"] == pytest.approx(1.0, rel=0, abs=1e-12)
    assert metrics["mean_dice_hard"] == pytest.approx(1.0, rel=0, abs=1e-12)
    assert metrics["miou_metric"] == pytest.approx(1.0, rel=0, abs=1e-12)
