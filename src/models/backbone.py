"""2D vision backbone feature extractor for 2.5D MRI slabs.

Wraps timm architectures to extract fixed-dimensional spatial embeddings from
contiguous 3-channel slice slabs with batch-sequence folding and unfolding.
"""

from typing import Union
import timm
import torch
import torch.nn as nn


class SliceBackbone(nn.Module):
    """Modular 2D slice feature extractor wrapping timm vision backbones."""

    def __init__(
        self,
        model_name: str = "resnet34d",
        pretrained: bool = False,
        in_chans: int = 3,
        drop_rate: float = 0.0,
    ):
        """Initialize the slice backbone.

        Args:
            model_name: Architecture name recognized by timm (e.g., 'resnet34d', 'convnext_tiny').
            pretrained: Whether to load ImageNet pretrained weights.
            in_chans: Number of input channels (typically 3 for contiguous 2.5D slabs).
            drop_rate: Classifier dropout rate.
        """
        super().__init__()
        self.model_name = model_name
        self.in_chans = in_chans

        # num_classes=0 removes the final linear classification layer and returns pooled features
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )

        # Retrieve feature dimension
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            self.feature_dim = self.backbone.head.in_features
        else:
            # Probe with dummy input if attribute not directly exposed
            with torch.no_grad():
                dummy = torch.zeros(1, in_chans, 64, 64)
                out = self.backbone(dummy)
                self.feature_dim = out.shape[-1]

    @property
    def out_features(self) -> int:
        """Return the feature dimension size."""
        return self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            x: Input tensor, either 5D shape (B, S, C, H, W) or 4D shape (N, C, H, W).

        Returns:
            Extracted features: (B, S, D) if 5D input, or (N, D) if 4D input.
        """
        if x.dim() == 5:
            # (Batch, Slabs, Channels, Height, Width)
            B, S, C, H, W = x.shape
            x_flat = x.view(B * S, C, H, W)
            features = self.backbone(x_flat)
            # Ensure 2D before unfolding: (B * S, D)
            if features.dim() > 2:
                features = torch.flatten(features, 1)
            return features.view(B, S, -1)
        elif x.dim() == 4:
            # (Batch, Channels, Height, Width)
            features = self.backbone(x)
            if features.dim() > 2:
                features = torch.flatten(features, 1)
            return features
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, but got shape {tuple(x.shape)}")
