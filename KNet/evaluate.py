"""
evaluate.py

This module provides evaluation utilities for cine MRI segmentation models.
It computes quantitative metrics on validation or test datasets to assess
model performance.

Supported metrics include:
- Cross-entropy loss (averaged per image).
- Dice coefficient (both soft Dice from loss and hard Dice from predictions).
- Mean Intersection-over-Union (mIoU).
- Pixel accuracy and confusion matrix.
- Average Surface Distance (ASD) per class and overall, based on predicted
  and ground-truth segmentation boundaries.

These evaluation tools are designed to align with the training/validation
semantics defined in the Trainer and provide detailed performance summaries
for each class as well as overall model performance.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

from trainer import metric_mIoU
from utils.main_utils import dice_loss


@torch.no_grad()
def metric_hard_dice(logits: torch.Tensor, target: torch.Tensor, exclude_bg: bool = False) -> float:
    """
    Compute mean hard Dice score from segmentation logits.

    Returns
    -------
    float
        Mean Dice score across the batch (averaged over classes and samples).
    """
    pred = logits.argmax(1)   # [B,H,W]
    C = logits.size(1)
    B = pred.size(0)

    tot, cnt = 0.0, 0
    for b in range(B):
        s, k = 0.0, 0
        for c in range(C):
            if exclude_bg and c == 0:
                continue
            tp = ((pred[b] == c) & (target[b] == c)).sum().item()
            p  = (pred[b] == c).sum().item()
            g  = (target[b] == c).sum().item()
            denom = p + g
            if denom == 0:
                continue
            s += (2.0 * tp) / (denom + 1e-12)
            k += 1
        if k > 0:
            tot += s / k
            cnt += 1
    return tot / max(cnt, 1)


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
    print_per_class: bool = True
) -> Dict[str, float]:
    """
    Metrics computed to MATCH your training/validation code semantics:
      - Cross-Entropy: per-image mean, then averaged over images (same as your validate()).
      - Mean Dice: computed via your soft Dice (Dice = 1 - dice_loss), then averaged over images.
      - mIoU: uses your metric_mIoU(logits, seg) and averages across images (weighted by batch size).
      - Pixel Accuracy + Confusion Matrix: from hard predictions (argmax).
      - ASD: symmetric average surface distance per class (2D), averaged over images; units from voxel_spacing.
    """

    model.eval()

    # helpers (same batch format as your Trainer)
    def _unpack_batch(batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            return batch  # (k_us, seg, k_full)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            k_us, seg = batch
            return k_us, seg, k_us
        else:
            raise ValueError("Unexpected batch format.")

    # Confusion matrix utils (rows=true, cols=pred)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    def _update_confusion(pred_np: np.ndarray, gt_np: np.ndarray):
        mask = (gt_np >= 0) & (gt_np < num_classes)
        lab = num_classes * gt_np[mask].astype(np.int64) + pred_np[mask].astype(np.int64)
        counts = np.bincount(lab.ravel(), minlength=num_classes * num_classes)
        conf_mat[:] += counts.reshape(num_classes, num_classes)

    # ASD helpers (simple 8-neighborhood boundary; symmetric mean)
    def _binary_edges(mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)
        p = np.pad(mask, 1, mode='constant', constant_values=0)
        neighbors = (
            (p[0:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, 0:-2] & p[1:-1, 2:] &
             p[0:-2, 0:-2] & p[0:-2, 2:] & p[2:, 0:-2] & p[2:, 2:])
        )
        interior = (mask.astype(bool) & neighbors.astype(bool))
        edges = mask.astype(bool) & (~interior)
        return np.argwhere(edges).astype(np.float32)

    def _asd_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: Tuple[float, float]) -> float:
        A = _binary_edges(pred_bin)
        B = _binary_edges(gt_bin)
        if A.shape[0] == 0 or B.shape[0] == 0:
            return np.nan
        dy, dx = spacing
        # A->B
        dyAB = (A[:, None, 0] - B[None, :, 0]) * dy
        dxAB = (A[:, None, 1] - B[None, :, 1]) * dx
        dAB = np.sqrt(dyAB**2 + dxAB**2).min(axis=1).mean()
        # B->A
        dyBA = (B[:, None, 0] - A[None, :, 0]) * dy
        dxBA = (B[:, None, 1] - A[None, :, 1]) * dx
        dBA = np.sqrt(dyBA**2 + dxBA**2).min(axis=1).mean()
        return 0.5 * (dAB + dBA)

    # accumulators aligned with your validate()
    # CE/Dice/mIoU: average per image (batch-weighted)
    ce_sum = 0.0
    dice_sum = 0.0
    hard_dice_sum = 0.0
    miou_sum = 0.0
    img_count = 0

    # ASD: collect per-class values across images
    asd_per_class: Dict[int, List[float]] = {c: [] for c in range(num_classes)}

    for batch in val_loader:
        k_us, seg, _ = _unpack_batch(batch)
        k_us = k_us.to(device, non_blocking=True)
        seg  = seg.to(device, non_blocking=True)

        logits = model(k_us)

        # Cross-entropy: per-image mean (your default), then image-averaged
        ce_batch = F.cross_entropy(logits, seg, reduction="mean").item()
        ce_sum += ce_batch * k_us.size(0)

        # Soft Dice (your dice_loss): Dice = 1 - dice_loss
        # reuse your dice_loss function already defined in the script
        dice_coeff_batch = (1.0 - dice_loss(logits, seg).item())
        dice_sum += dice_coeff_batch * k_us.size(0)
        hd_batch = metric_hard_dice(logits, seg, exclude_bg=False)
        hard_dice_sum += hd_batch * k_us.size(0)

        # mIoU: exactly your function & averaging semantics
        miou_batch = metric_mIoU(logits, seg)
        miou_sum += miou_batch * k_us.size(0)

        # Confusion matrix & ASD need hard predictions
        pred = logits.argmax(1)  # [B,H,W]
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        gt_np   = seg.detach().cpu().numpy().astype(np.int64)
        _update_confusion(pred_np, gt_np)

        # ASD per class per image (if you prefer to exclude background here, start from c=1)
        for b in range(pred_np.shape[0]):
            for c in range(num_classes):
                asd_val = _asd_binary((pred_np[b] == c).astype(np.uint8),
                                      (gt_np[b]   == c).astype(np.uint8),
                                      voxel_spacing)
                if not np.isnan(asd_val):
                    asd_per_class[c].append(asd_val)

        img_count += k_us.size(0)

    # Final aggregates (exactly like your validate printing style)
    avg_ce   = ce_sum   / max(img_count, 1)
    avg_dice = dice_sum / max(img_count, 1)
    avg_hard_dice = hard_dice_sum / max(img_count, 1)
    avg_miou = miou_sum / max(img_count, 1)

    # Pixel accuracy from confusion matrix
    total = conf_mat.sum()
    correct = np.trace(conf_mat)
    pixel_acc = float(correct) / float(total + 1e-12)

    # Per-class IoU/Dice from confusion (for display; may differ from "soft" numbers)
    TP = np.diag(conf_mat).astype(np.float64)
    FN = conf_mat.sum(axis=1) - TP
    FP = conf_mat.sum(axis=0) - TP
    iou_c  = TP / (TP + FP + FN + 1e-12)
    dice_c = (2 * TP) / (2 * TP + FP + FN + 1e-12)

    # ASD summaries
    asd_mean_per_class = {c: (float(np.mean(v)) if len(v) > 0 else np.nan)
                          for c, v in asd_per_class.items()}
    overall_asd_vals = [v for v in asd_mean_per_class.values() if not np.isnan(v)]
    overall_asd = float(np.mean(overall_asd_vals)) if len(overall_asd_vals) > 0 else np.nan

    # Print (aligned)
    print("=== Segmentation Evaluation (Aligned with training semantics) ===")
    print(f"Pixel Accuracy : {pixel_acc:.4f}")
    print(f"Cross Entropy  : {avg_ce:.6f}   (per-image mean, image-averaged)")
    # print(f"Mean Dice (soft): {avg_dice:.4f} (Dice = 1 - dice_loss)")
    print(f"Mean Dice: {avg_hard_dice:.4f} (argmax, image-averaged)")
    print(f"Mean IoU (mIoU): {avg_miou:.4f} (via metric_mIoU, image-averaged)")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(conf_mat)

    if print_per_class:
        print("\nPer-class (from hard predictions) IoU / Dice:")
        for c in range(num_classes):
            print(f"  Class {c:02d}: IoU={iou_c[c]:.4f}  Dice={dice_c[c]:.4f}")

        print("\nPer-class ASD (units match voxel_spacing):")
        for c in range(num_classes):
            v = asd_mean_per_class[c]
            print(f"  Class {c:02d}: ASD={(f'{v:.4f}' if not np.isnan(v) else 'nan')}")

    print(f"\nOverall ASD (mean over classes): {(f'{overall_asd:.4f}' if not np.isnan(overall_asd) else 'nan')}")
    print("=================================================================\n")

    return {
        "pixel_accuracy": pixel_acc,
        "cross_entropy": avg_ce,
        "mean_dice_hard": avg_hard_dice,
        "miou_metric": avg_miou,
        "overall_asd": overall_asd,
    }


