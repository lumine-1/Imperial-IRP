"""
visualise_utils.py

Visualisation utils for this project.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def ifft2c_torch(x: torch.Tensor) -> torch.Tensor:
    """Apply centered 2D inverse FFT with orthogonal normalization."""
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def compute_metrics(pred: np.ndarray, gt: np.ndarray, n_class: int = 4):
    """Compute segmentation metrics including Pixel Accuracy, mIoU, and Dice."""
    # flatten
    pred = pred.flatten()
    gt   = gt.flatten()

    mask = (gt >= 0) & (gt < n_class)
    cm = confusion_matrix(gt[mask], pred[mask], labels=list(range(n_class)))

    # calculate scores
    intersection = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = intersection / (union + 1e-8)
    miou = iou.mean()
    pixel_acc = (pred == gt).sum() / len(gt)
    dice = 2 * intersection / (cm.sum(1) + cm.sum(0) + 1e-8)
    mdice = dice.mean()

    return {
        "Pixel Acc": pixel_acc,
        "mIoU": miou,
        "Mean Dice": mdice,
        "IoU per class": iou,
        "Dice per class": dice,
    }


def _ri_to_cplx(k_ri: torch.Tensor) -> torch.Tensor:
    """Convert real/imaginary channel representation into complex-valued tensor."""
    if k_ri.dim() == 3:  # [2C,H,W]
        C = k_ri.shape[0] // 2
        k2 = k_ri.view(C, 2, *k_ri.shape[-2:])
        return torch.complex(k2[:, 0], k2[:, 1])
    elif k_ri.dim() == 4:  # [B,2C,H,W]
        B, CC, H, W = k_ri.shape
        C = CC // 2
        k2 = k_ri.view(B, C, 2, H, W)
        return torch.complex(k2[:, :, 0], k2[:, :, 1])
    else:
        raise ValueError(f"Unexpected k_ri shape: {k_ri.shape}")


def _zf_image_from_ri(k_ri: torch.Tensor) -> np.ndarray:
    """Reconstruct a zero-filled RSS image from RI-encoded k-space."""
    kc = _ri_to_cplx(k_ri)
    if kc.dim() == 2:
        img = ifft2c_torch(kc.unsqueeze(0)).abs().squeeze(0)
        return img.detach().cpu().numpy()
    else:
        img_c = ifft2c_torch(kc)
        mag = torch.abs(img_c)
        rss = torch.sqrt((mag ** 2).sum(dim=0))
        return rss.detach().cpu().numpy()


def _estimate_R_and_ky_mask(k_ri: torch.Tensor, eps: float = 0.0):
    """Estimate acceleration factor and ky sampling mask from RI-encoded k-space."""
    C = k_ri.shape[0] // 2
    H, W = k_ri.shape[-2], k_ri.shape[-1]
    a = k_ri.view(C, 2, H, W).abs().sum(dim=1)
    ky_keep = (a.sum(dim=0).sum(dim=-1) > eps)
    lines = int(ky_keep.sum().item())
    R_est = (H / max(lines, 1)) if lines > 0 else float("inf")
    return float(R_est), ky_keep.cpu().numpy()


def visualize_single_sample(model, dataset: Dataset, index: int, device: torch.device, show_mask_lines: bool = True):
    """
    Visualize model prediction on a single sample from a k-space dataset.

    Supports both dataset formats:
      - Undersampled: (k_us_ri, seg, k_full_ri)
      - Fully sampled (legacy): (k_ri, seg)

    Shows:
      - Zero-filled reconstruction from undersampled k-space
      - (Optional) zero-filled fully sampled image
      - Model prediction
      - Ground truth segmentation
    """
    model.eval()
    sample = dataset[index]
    if isinstance(sample, (list, tuple)) and len(sample) == 3:
        k_us_ri, seg_gt, k_full_ri = sample
    elif isinstance(sample, (list, tuple)) and len(sample) == 2:
        k_us_ri, seg_gt = sample
        k_full_ri = None
    else:
        raise ValueError("Unexpected dataset item, expect (k, seg) or (k_us, seg, k_full).")

    # reference
    k_in = k_us_ri.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(k_in)
        pred = logits.argmax(1).squeeze(0).cpu().numpy()

    # show image
    zf_us   = _zf_image_from_ri(k_us_ri)
    R_est, ky_keep = _estimate_R_and_ky_mask(k_us_ri)
    zf_full = _zf_image_from_ri(k_full_ri) if k_full_ri is not None else None

    # plot
    cols = 4 if zf_full is not None else 3
    fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 4))

    # zero filling for under-sampling
    axs[0].imshow(zf_us, cmap="gray")
    axs[0].set_title(f"ZF (undersampled, ~R={R_est:.1f})")
    if show_mask_lines:
        ys = np.where(ky_keep)[0]
        for y in ys:
            axs[0].axhline(y, color="lime", alpha=0.10, linewidth=0.5)

    # zero filling
    if zf_full is not None:
        axs[1].imshow(zf_full, cmap="gray")
        axs[1].set_title("ZF (fully sampled)")

    # prediction and GT
    ax_pred = axs[2] if zf_full is not None else axs[1]
    ax_gt   = axs[3] if zf_full is not None else axs[2]
    ax_pred.imshow(pred, cmap="tab10", vmin=0)
    ax_pred.set_title("Prediction")
    ax_gt.imshow(seg_gt.numpy(), cmap="tab10", vmin=0)
    ax_gt.set_title("Ground Truth")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # print metrix
    try:
        metrics = compute_metrics(pred, seg_gt.numpy(), n_class=model.cfg.num_classes if hasattr(model, "cfg") else logits.size(1))
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {np.round(v, 3)}")
            else:
                print(f"{k}: {v:.4f}")
    except NameError:
        pass


@torch.no_grad()
def visualize_image_sample(
    model: torch.nn.Module,
    dataset,
    index: int = 0,
    device: torch.device = torch.device("cpu"),
    alpha: float = 0.8,
    class_names=None,
    save_path: str | None = None
):
    """
    Visualize segmentation prediction on a single image sample.

    Shows:
      - Input grayscale image
      - Image with ground truth mask overlay
      - Image with model prediction overlay
    """
    model.eval()

    # select a sample
    img, msk = dataset[index]
    H, W = img.shape[-2], img.shape[-1]
    n_classes = getattr(dataset, "n_classes", None)
    if n_classes is None:
        tmp = model(img.unsqueeze(0).to(device))
        n_classes = tmp.shape[1]
    x = img.unsqueeze(0).to(device)
    logits = model(x)
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int64)
    img_np = img.squeeze(0).cpu().numpy()
    gt_np  = msk.cpu().numpy().astype(np.int64)

    # colors
    base_colors = [
        (0, 0, 0, 0.0),
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    if n_classes > len(base_colors):
        extra = n_classes - len(base_colors)
        base_colors += base_colors[1:1+extra]
    cmap = ListedColormap(base_colors[:n_classes])
    norm = BoundaryNorm(list(range(n_classes+1)), ncolors=n_classes)

    # figure labels
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    legend_handles = [mpatches.Patch(color=base_colors[i], label=class_names[i])
                      for i in range(1, n_classes)]

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    ax0, ax1, ax2 = axes

    # original image
    ax0.imshow(img_np, cmap="gray")
    ax0.set_title("Image")
    ax0.axis("off")

    # GT overlap
    ax1.imshow(img_np, cmap="gray")
    ax1.imshow(gt_np, cmap=cmap, norm=norm, alpha=(gt_np > 0) * alpha)
    ax1.set_title("Image + GT")
    ax1.axis("off")

    # prediction overlap
    ax2.imshow(img_np, cmap="gray")
    ax2.imshow(pred, cmap=cmap, norm=norm, alpha=(pred > 0) * alpha)
    ax2.set_title("Image + Pred")
    ax2.axis("off")
    if len(legend_handles) > 0:
        fig.legend(handles=legend_handles, loc="lower center", ncol=min(5, len(legend_handles)),
                   bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
