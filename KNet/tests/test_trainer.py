import torch
import trainer as TR
from models.image_model import ImageSegModel
from models.k_image_model import KImgSegModel2
from models.k_model import KSegModel, ModelCfg


# metric_mIoU tests
def test_metric_miou_perfect_match():
    B, C, H, W = 2, 3, 8, 10
    mask = torch.randint(0, C, (B, H, W))
    logits = torch.zeros(B, C, H, W)
    for b in range(B):
        logits[b, mask[b], torch.arange(H).unsqueeze(1), torch.arange(W)] = 1.0
    miou = TR.metric_mIoU(logits, mask)
    assert abs(miou - 1.0) < 1e-6

def test_metric_miou_ignores_missing_class():
    B, C, H, W = 1, 3, 4, 5
    # mask only has class 0
    mask = torch.zeros(B, H, W, dtype=torch.long)
    logits = torch.zeros(B, C, H, W)
    logits[:, 0] = 10.0
    miou = TR.metric_mIoU(logits, mask)
    assert abs(miou - 1.0) < 1e-6


def test_ksegmodel_forward():
    model = KSegModel(ModelCfg())
    x = torch.randn(2, 2, 192, 448)
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)


def test_imagesegmodel_forward():
    model = ImageSegModel(n_classes=4)
    x = torch.randn(2, 1, 192, 448)
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)


def test_kimgsegmodel_forward():
    model = KImgSegModel2(k_in_ch=2, num_classes=4)
    x = torch.randn(2, 3, 192, 448)
    logits = model(x)
    assert logits.shape == (2, 4, 192, 448)



