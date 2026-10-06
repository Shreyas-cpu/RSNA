"""Gated Multiple Instance Learning (MIL) attention pooling engine.

Implements Ilse et al. (2018) Gated Attention pooling to dynamically weight focal
pathology slices and collapse variable-length slice sequences into fixed study embeddings.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class GatedAttentionPool(nn.Module):
    """Gated Attention Multiple Instance Learning (MIL) pooling module."""

    def __init__(self, in_features: int, hidden_features: int = 128):
        """Initialize the gated attention pool.

        Args:
            in_features: Input feature dimension per slice (D).
            hidden_features: Dimension of internal projection space (L).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        # Gated branches: tanh branch (V) and sigmoid gating branch (U)
        self.linear_v = nn.Linear(in_features, hidden_features)
        self.linear_u = nn.Linear(in_features, hidden_features)
        # Final projection to scalar attention score
        self.linear_w = nn.Linear(hidden_features, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate sequence of slice features into a single pooled representation.

        Args:
            x: Sequence tensor of shape (Batch, Slabs, Dimension) [B, S, D].
            mask: Optional binary mask of shape (B, S) where 1=valid, 0=padded.

        Returns:
            Tuple of:
                - pooled: Fixed-size study representation of shape (Batch, Dimension) [B, D].
                - weights: Normalized attention weights of shape (Batch, Slabs, 1) [B, S, 1].
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor of shape (B, S, D), got {tuple(x.shape)}")

        # 1. Compute gated attention scores: w^T * (tanh(V * x) * sigmoid(U * x))
        v = torch.tanh(self.linear_v(x))
        u = torch.sigmoid(self.linear_u(x))
        scores = self.linear_w(v * u)  # Shape: (B, S, 1)

        # 2. Apply sequence padding mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # (B, S, 1)
            scores = scores.masked_fill(~mask.bool(), -1e9)

        # 3. Softmax normalization over slice sequence dimension (dim=1)
        weights = torch.softmax(scores, dim=1)  # Shape: (B, S, 1)

        # 4. Weighted combination: sum over slices
        pooled = torch.sum(weights * x, dim=1)  # Shape: (B, D)

        return pooled, weights
