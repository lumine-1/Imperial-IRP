import pytest
import torch
from models.k_model import KSegModel, ModelCfg
from models.image_model import ImageSegModel
from models.k_image_model import KImgSegModel2



def test_ksegmodel_forward():
    model = KSegModel(ModelCfg())
    x = torch.randn(2, 2, 192, 448)
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)


def test_ksegmodel_forward_and_loss():
    model = KSegModel(ModelCfg())
    # fake k-space
    x = torch.randn(2, 2, 192, 448)
    # fake segmentation mask
    mask = torch.randint(0, 4, (2, 192, 448))
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)

    # check loss dict keys
    losses = model.loss(logits, mask, x)
    assert "total" in losses and losses["total"].requires_grad
    # backward
    losses["total"].backward()


def test_imagesegmodel_forward():
    model = ImageSegModel(n_classes=4)
    x = torch.randn(2, 1, 192, 448)
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)


def test_imagesegmodel_forward_and_loss():
    model = ImageSegModel(n_classes=4)
    x = torch.randn(2, 1, 192, 448)
    target = torch.randint(0, 4, (2, 192, 448))
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)

    losses = model.loss(logits, target)
    assert "total" in losses and losses["total"].requires_grad
    losses["total"].backward()


def test_kimgsegmodel_forward():
    model = KImgSegModel2(k_in_ch=2, num_classes=4)
    x = torch.randn(2, 3, 192, 448)
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)


def test_kimgsegmodel_forward_and_loss():
    model = KImgSegModel2(k_in_ch=2, num_classes=4)
    x = torch.randn(2, 3, 192, 448)   # k-space(2) + image(1)
    target = torch.randint(0, 4, (2, 192, 448))
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)

    losses = model.loss(logits, target)
    assert "total" in losses and losses["total"].requires_grad
    losses["total"].backward()


def test_invalid_input_shape_raises():
    model = KImgSegModel2(k_in_ch=2, num_classes=4)
    bad_x = torch.randn(2, 2, 192, 448)  # missing image channel
    with pytest.raises(AssertionError):
        _ = model(bad_x)


