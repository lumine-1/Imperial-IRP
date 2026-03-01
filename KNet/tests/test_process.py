import torch
import pytest
from models.k_model import KSegModel, ModelCfg
from models.image_model import ImageSegModel
from models.k_image_model import KImgSegModel2

H, W = 192, 448
NC = 4


def test_ksegmodel_forward_and_loss_backward():
    model = KSegModel(ModelCfg(in_ch=2, num_classes=NC, unet_base=24))
    x = torch.randn(2, 2, H, W)
    y = torch.randint(0, NC, (2, H, W))
    logits = model(x)
    assert logits.shape == (2, NC, H, W)
    losses = model.loss(logits, y, x)
    assert "total" in losses and losses["total"].requires_grad
    losses["total"].backward()


def test_imagesegmodel_forward_and_loss_backward():
    model = ImageSegModel(n_classes=NC, base=24)
    x = torch.randn(2, 1, H, W)
    y = torch.randint(0, NC, (2, H, W))
    logits = model(x)
    assert logits.shape == (2, NC, H, W)
    losses = model.loss(logits, y)
    assert "total" in losses and losses["total"].requires_grad
    losses["total"].backward()


def test_kimgsegmodel_forward_and_loss_backward():
    model = KImgSegModel2(k_in_ch=2, num_classes=NC, unet_base=24)
    x = torch.randn(2, 3, H, W)
    y = torch.randint(0, NC, (2, H, W))
    logits = model(x)
    assert logits.shape == (2, NC, H, W)
    losses = model.loss(logits, y)
    assert "total" in losses and losses["total"].requires_grad
    losses["total"].backward()


def test_kimgsegmodel_bad_input_raises():
    model = KImgSegModel2(k_in_ch=2, num_classes=NC)
    bad = torch.randn(2, 2, H, W)
    with pytest.raises(AssertionError):
        _ = model(bad)


