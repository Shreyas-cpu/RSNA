"""Unit tests for MultiPlaneFusion and MultiPlaneKneeModel."""

import numpy as np
import pytest
import torch

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.models.multi_plane import MultiPlaneFusion, MultiPlaneKneeModel


@pytest.fixture
def dummy_planes():
    """Generates synthetic 5D tensors (B=2, S=3, C=3, H=32, W=32) for 3 planes."""
    return {
        "sagittal": torch.randn(2, 3, 3, 32, 32),
        "coronal": torch.randn(2, 4, 3, 32, 32),  # Variable slices S=4
        "axial": torch.randn(2, 2, 3, 32, 32),     # Variable slices S=2
    }


def test_multi_plane_fusion_shape():
    """Verify MultiPlaneFusion produces (B, D) fused representation from 3 plane vectors."""
    B, D = 4, 64
    fusion = MultiPlaneFusion(feature_dim=D, num_heads=2)

    plane_feats = {
        "sagittal": torch.randn(B, D),
        "coronal": torch.randn(B, D),
        "axial": torch.randn(B, D),
    }

    out = fusion(plane_feats)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()


def test_multi_plane_fusion_missing_plane():
    """Verify MultiPlaneFusion handles omitted planes without errors or NaNs."""
    B, D = 3, 64
    fusion = MultiPlaneFusion(feature_dim=D, num_heads=2)

    # Axial series omitted
    partial_feats = {
        "sagittal": torch.randn(B, D),
        "coronal": torch.randn(B, D),
    }

    out = fusion(partial_feats)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()


def test_multi_plane_model_forward_and_predict_proba(dummy_planes):
    """Verify end-to-end MultiPlaneKneeModel forward pass and probability predictions."""
    model = MultiPlaneKneeModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_classes=len(DEFAULT_TARGET_COLS),
        fusion_heads=2,
    )
    model.eval()

    # Forward pass
    logits = model(dummy_planes)
    assert logits.shape == (2, len(DEFAULT_TARGET_COLS))

    # Test attention return
    logits_attn, attn_dict = model(dummy_planes, return_attention=True)
    assert "sagittal" in attn_dict
    assert "coronal" in attn_dict
    assert "axial" in attn_dict

    # Predict proba
    probs = model.predict_proba(dummy_planes)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, len(DEFAULT_TARGET_COLS))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_multi_plane_model_missing_plane():
    """Verify MultiPlaneKneeModel executes gracefully when a plane is missing."""
    model = MultiPlaneKneeModel(
        backbone_name="resnet34d",
        pretrained=False,
        fusion_heads=2,
    )
    model.eval()

    # Only sagittal and coronal present
    partial_dict = {
        "sagittal": torch.randn(2, 3, 3, 32, 32),
        "coronal": torch.randn(2, 3, 3, 32, 32),
    }

    logits = model(partial_dict)
    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_multi_plane_model_backward_and_amp(dummy_planes):
    """Verify backward gradient flow and AMP compatibility."""
    model = MultiPlaneKneeModel(
        backbone_name="resnet34d",
        pretrained=False,
        fusion_heads=2,
    )
    model.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in dummy_planes.items()}

    with torch.autocast(device_type=device, enabled=True):
        logits = model(inputs)
        loss = logits.sum()

    loss.backward()

    # Verify gradients exist on backbone, fusion, and head
    assert model.classifier.weight.grad is not None
    assert model.fusion.proj[0].weight.grad is not None
