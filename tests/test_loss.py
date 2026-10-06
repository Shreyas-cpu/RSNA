import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.loss import compute_pos_weights, create_loss_fn


def test_compute_pos_weights_rare_vs_common():
    """Verify rare positive classes receive significantly higher weights than common ones."""
    # 100 samples: class 0 has 5% positives (rare), class 1 has 50% positives (balanced)
    y = np.zeros((100, 2), dtype=float)
    y[:5, 0] = 1.0   # 5 pos, 95 neg -> sqrt(95/5) = sqrt(19) ~ 4.36
    y[:50, 1] = 1.0  # 50 pos, 50 neg -> sqrt(50/50) = 1.0

    weights = compute_pos_weights(y, clip_range=(1.0, 10.0))

    assert weights.shape == (2,)
    assert pytest.approx(weights[0].item(), rel=1e-2) == 4.36
    assert pytest.approx(weights[1].item(), rel=1e-2) == 1.0
    assert weights[0] > weights[1]


def test_compute_pos_weights_clamping():
    """Verify clipping bounds prevent extreme gradient scaling."""
    # Class with 0 positives
    y = np.zeros((100, 1), dtype=float)
    weights = compute_pos_weights(y, clip_range=(1.0, 10.0))

    assert weights[0].item() == 10.0


def test_create_loss_fn_backward():
    """Verify create_loss_fn produces valid loss and propagates gradients correctly."""
    pos_weights = torch.tensor([2.0, 5.0], dtype=torch.float32)
    criterion = create_loss_fn(pos_weights=pos_weights)

    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ], dtype=torch.float32)

    loss = criterion(logits, targets)
    loss.backward()

    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
