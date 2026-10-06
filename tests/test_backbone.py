import pytest
import torch
from src.models.backbone import SliceBackbone


def test_backbone_5d_input_resnet():
    """Verify SliceBackbone handles 5D tensor (B, S, C, H, W) and returns (B, S, D)."""
    backbone = SliceBackbone(model_name="resnet34d", pretrained=False, in_chans=3)
    x = torch.randn(2, 4, 3, 64, 64)

    out = backbone(x)

    assert out.shape == (2, 4, 512)
    assert backbone.feature_dim == 512
    assert backbone.out_features == 512


def test_backbone_4d_input():
    """Verify SliceBackbone handles direct 4D tensor (N, C, H, W)."""
    backbone = SliceBackbone(model_name="resnet34d", pretrained=False, in_chans=3)
    x = torch.randn(3, 3, 64, 64)

    out = backbone(x)
    assert out.shape == (3, 512)


def test_backbone_efficientnet():
    """Verify SliceBackbone works with EfficientNet architectures."""
    backbone = SliceBackbone(model_name="efficientnet_b0", pretrained=False, in_chans=3)
    x = torch.randn(2, 3, 3, 64, 64)

    out = backbone(x)
    assert out.shape == (2, 3, 1280)
    assert backbone.feature_dim == 1280


def test_backbone_backward_gradients():
    """Verify gradients propagate through the backbone."""
    backbone = SliceBackbone(model_name="resnet34d", pretrained=False, in_chans=3)
    x = torch.randn(2, 2, 3, 64, 64, requires_grad=True)

    out = backbone(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Verify model parameters received gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
    assert has_grads
