"""End-to-end 2.5D Multi-Slice Knee Abnormality Model.

Combines a 2D vision backbone (timm), Gated Attention sequence pooling (MIL),
and a 12-class multi-label classification head predicting continuous probabilities.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.attention import GatedAttentionPool
from src.models.backbone import SliceBackbone


class RSNAKneeModel(nn.Module):
    """End-to-end 2.5D vision model with Gated Attention pooling for knee MRI abnormalities."""

    def __init__(
        self,
        backbone_name: str = "resnet34d",
        num_classes: int = 12,
        pretrained: bool = False,
        in_chans: int = 3,
        hidden_features: int = 128,
        drop_rate: float = 0.2,
    ):
        """Initialize the model.

        Args:
            backbone_name: Vision backbone name supported by timm.
            num_classes: Number of abnormality target classes (default 12).
            pretrained: Whether to load ImageNet pretrained weights.
            in_chans: Number of channels per slice slab (default 3 for 2.5D triplets).
            hidden_features: Dimension of internal gated attention projection.
            drop_rate: Classifier dropout probability.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # 1. 2D Slice feature extractor
        self.backbone = SliceBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )

        # 2. Gated Attention MIL sequence pooler
        self.pool = GatedAttentionPool(
            in_features=self.backbone.feature_dim,
            hidden_features=hidden_features,
        )

        # 3. 12-Class multi-label classification head
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, S, C, H, W).
            mask: Optional sequence mask (B, S).
            return_attention: If True, returns (logits, attention_weights).

        Returns:
            Logits of shape (B, num_classes) or tuple of (logits, weights).
        """
        # 1. Extract feature sequence: (B, S, D)
        features = self.backbone(x)

        # 2. Pool across slice sequence: (B, D) and attention weights (B, S, 1)
        pooled, weights = self.pool(features, mask=mask)

        # 3. Linear prediction head: (B, num_classes)
        logits = self.classifier(self.dropout(pooled))

        if return_attention:
            return logits, weights
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute continuous calibrated probabilities in [0.0, 1.0] for all classes.

        Args:
            x: Input tensor of shape (B, S, C, H, W).
            mask: Optional sequence mask.

        Returns:
            Continuous probabilities of shape (B, num_classes) bounded in [0.0, 1.0].
        """
        self.eval()
        logits = self.forward(x, mask=mask, return_attention=False)
        return torch.sigmoid(logits)
