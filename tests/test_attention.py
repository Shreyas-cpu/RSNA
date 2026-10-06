import pytest
import torch
from src.models.attention import GatedAttentionPool


def test_attention_pool_shape_and_sum():
    """Verify GatedAttentionPool collapses (B, S, D) to (B, D) with weights summing to 1.0."""
    pool = GatedAttentionPool(in_features=64, hidden_features=32)
    x = torch.randn(4, 8, 64)

    pooled, weights = pool(x)

    assert pooled.shape == (4, 64)
    assert weights.shape == (4, 8, 1)

    # Weights must sum to 1.0 along sequence dimension
    sums = weights.sum(dim=1).squeeze(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_attention_pool_masking():
    """Verify masked positions receive zero attention weight."""
    pool = GatedAttentionPool(in_features=32, hidden_features=16)
    x = torch.randn(2, 6, 32)
    # Mask out the last 2 slices of each item
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ], dtype=torch.bool)

    pooled, weights = pool(x, mask=mask)

    assert pooled.shape == (2, 32)
    # Check that masked positions have essentially zero weight (< 1e-4)
    assert weights[0, 4, 0].item() < 1e-4
    assert weights[0, 5, 0].item() < 1e-4
    assert weights[1, 3, 0].item() < 1e-4

    # Valid positions still sum to 1.0
    assert pytest.approx(weights[0].sum().item(), rel=1e-4) == 1.0
    assert pytest.approx(weights[1].sum().item(), rel=1e-4) == 1.0


def test_attention_pool_backward():
    """Verify gradients propagate through attention projections."""
    pool = GatedAttentionPool(in_features=48, hidden_features=24)
    x = torch.randn(2, 5, 48, requires_grad=True)

    pooled, weights = pool(x)
    loss = pooled.sum()
    loss.backward()

    assert x.grad is not None
    assert pool.linear_w.weight.grad is not None
