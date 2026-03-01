"""
trainer.py

Training/validation utilities for cine MRI segmentation and reconstruction.
This module wraps a PyTorch model, optimizer, and dataloaders to run full
epochs with mixed precision, gradient scaling, and simple checkpointing.

Key features
------------
- Supports two batch formats: (k_us, seg) and (k_us, seg, k_full).
- Computes and logs CE loss, Dice loss, optional reconstruction loss, and mIoU.
- Uses AdamW and tqdm progress bars.
- Saves the best model by validation mIoU to the configured output path.
"""

from __future__ import annotations
from pathlib import Path
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW


@torch.no_grad()
def metric_mIoU(logits: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(1)
    miou_tot, classes = 0.0, 0
    for c in range(logits.size(1)):
        inter = ((pred == c) & (mask == c)).sum()
        union = ((pred == c) | (mask == c)).sum()
        if union == 0:
            continue
        miou_tot += (inter / union).item()
        classes += 1
    return miou_tot / max(classes, 1)


class Trainer:
    """
    Training and validation loop for segmentation/reconstruction models.
    This class wraps the model, optimizer, and training/validation dataloaders,
    providing methods for a full training epoch and evaluation.

    Notes
    -------
    - Supports both 2-element batches `(k_us, seg)` and 3-element batches
      `(k_us, seg, k_full)` depending on whether reconstruction supervision
      is included.
    - Tracks and prints CE, Dice loss, and optional reconstruction loss.
    - Saves the best model based on validation mIoU.
    """
    def __init__(self, model, train_loader, val_loader, lr, out_dir: Path, dev, weight):
        self.model, self.train_loader, self.val_loader = model.to(dev), train_loader, val_loader
        self.opt, self.scaler, self.dev = AdamW(model.parameters(), lr, weight_decay=4e-4), GradScaler(), dev
        self.best = 0.0
        self.out_dir = out_dir
        self.weight = weight

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            k_us, seg, k_full = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            k_us, seg = batch
            k_full = k_us
        else:
            raise ValueError("Unexpected batch format.")
        return k_us, seg, k_full

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"train {epoch}")
        tot_ce = tot_dice = tot_recon = tot_total = 0.0
        cnt = 0

        for batch in pbar:
            k_us, seg, k_full = self._unpack_batch(batch)
            k_us = k_us.to(self.dev, non_blocking=True)
            seg = seg.to(self.dev, non_blocking=True)
            k_full = k_full.to(self.dev, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)
            with autocast():
                logits = self.model(k_us)
                losses = self.model.loss(logits, seg, k_full)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            # calculate training scores
            bsz = k_us.size(0)
            cnt += bsz
            tot_ce += losses["ce"].item() * bsz
            tot_dice += losses["dice"].item() * bsz
            tot_total += losses["total"].item() * bsz
            if "recon" in losses:
                tot_recon += losses["recon"].item() * bsz

            # show on the bar
            pbar.set_postfix(
                ce=f"{losses['ce'].item():.3f}",
                dice=f"{losses['dice'].item():.3f}",
                recon=f"{losses.get('recon', 0.0):.3f}",
                total=f"{losses['total'].item():.3f}",
            )

        n = max(cnt, 1)
        # printing results
        msg = f"CE={tot_ce/n:.4e}  DICE_LOSS={tot_dice/n:.4e}  TOTAL={tot_total/n:.4e}"
        # if tot_recon > 0:
        #     msg += f"  RECON={tot_recon/n:.4e}"
        print(msg)

    @torch.no_grad()
    # validation set
    def validate(self, epoch):
        self.model.eval()
        miou_sum = ce_sum = dice_sum = recon_sum = 0.0
        count = 0

        for batch in tqdm(self.val_loader, desc=f"val   {epoch}"):
            k_us, seg, k_full = self._unpack_batch(batch)
            k_us = k_us.to(self.dev, non_blocking=True)
            seg = seg.to(self.dev, non_blocking=True)
            k_full = k_full.to(self.dev, non_blocking=True)

            logits = self.model(k_us)
            loss_dict = self.model.loss(logits, seg, k_full)

            # sum up during the process
            miou_sum += metric_mIoU(logits, seg) * k_us.size(0)
            ce_sum += loss_dict["ce"].item() * k_us.size(0)
            dice_sum += loss_dict["dice"].item() * k_us.size(0)
            if "recon" in loss_dict:
                recon_sum += loss_dict["recon"].item() * k_us.size(0)
            count += k_us.size(0)

        # calculate scores
        n = max(count, 1)
        avg_miou = miou_sum / n
        avg_ce = ce_sum / n
        avg_dice = dice_sum / n
        avg_recon = recon_sum / n if recon_sum > 0 else 0.0

        # print results every loops
        print(f"[Epoch {epoch}]: mIoU={avg_miou:.4f}, CE={avg_ce:.4e}, Dice_loss={avg_dice:.4f}"
              # + (f", Recon={avg_recon:.4e}" if recon_sum > 0 else "")
              )
        print("")

        # save the best model
        if avg_miou > self.best:
            self.best = avg_miou
            self.out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.out_dir / self.weight)

        return avg_miou



