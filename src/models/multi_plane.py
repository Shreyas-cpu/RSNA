"""Multi-Plane Feature Fusion & RSNA Knee Vision Model.

Fuses Sagittal, Coronal, and Axial series representations via cross-plane
attention and gated projection for holistic 12-class study diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.models.attention import GatedAttentionPool
from src.models.backbone import SliceBackbone

PLANES_DEFAULT = ("sagittal", "coronal", "axial")
SLOTS_6_DEFAULT = (
    "sagittal_fs",
    "coronal_fs",
    "axial_fs",
    "sagittal_t1",
    "coronal_t1",
    "axial_t1",
)


def get_slot_name(plane: str, is_fluid_sensitive: bool = True) -> str:
    """Resolve canonical slot name from anatomical plane and contrast sensitivity."""
    clean_plane = plane.strip().lower()
    suffix = "fs" if is_fluid_sensitive else "t1"
    return f"{clean_plane}_{suffix}"


class MultiPlaneFusion(nn.Module):
    """Fuses multi-plane pooled representations with cross-plane self-attention."""

    def __init__(
        self,
        feature_dim: int,
        planes: Tuple[str, ...] = PLANES_DEFAULT,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize MultiPlaneFusion.

        Args:
            feature_dim: Channel dimension of plane embeddings.
            planes: Tuple of anatomical plane names.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.planes = list(planes)
        self.num_planes = len(self.planes)

        # Learnable plane positional embeddings
        self.plane_embeddings = nn.Parameter(
            torch.randn(self.num_planes, feature_dim) * 0.02
        )

        # Learnable missing plane token
        self.missing_token = nn.Parameter(
            torch.zeros(1, 1, feature_dim)
        )

        # Multihead self-attention across the planes
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Non-linear projection to unified study embedding
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * self.num_planes, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        plane_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse dictionary of plane embeddings into a single study representation.

        Args:
            plane_features: Dict mapping plane name to (B, D) tensor.

        Returns:
            Fused study embedding tensor of shape (B, D).
        """
        # Determine batch size and device from any available plane
        sample_tensor = next(iter(plane_features.values()))
        batch_size = sample_tensor.size(0)
        device = sample_tensor.device
        dtype = sample_tensor.dtype

        tokens_list: List[torch.Tensor] = []
        padding_mask_list: List[torch.Tensor] = []

        for p_idx, p_name in enumerate(self.planes):
            if p_name in plane_features and plane_features[p_name] is not None:
                p_feat = plane_features[p_name]  # (B, D)
                tokens_list.append(p_feat)
                padding_mask_list.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
            else:
                # Missing plane for this batch: use missing token
                missing = self.missing_token.expand(batch_size, 1, -1).squeeze(1).to(device=device, dtype=dtype)
                tokens_list.append(missing)
                # Mark as padded/missing
                padding_mask_list.append(torch.ones(batch_size, dtype=torch.bool, device=device))

        # Stack into (B, num_planes, D)
        tokens = torch.stack(tokens_list, dim=1)
        padding_mask = torch.stack(padding_mask_list, dim=1)  # (B, num_planes)

        # Add plane-specific positional embeddings
        tokens = tokens + self.plane_embeddings.unsqueeze(0).to(device=device, dtype=dtype)

        # Self-attention across planes
        # Note: if all planes are missing in a row, key_padding_mask causes NaN in standard softmax;
        # ensure at least one False exists or pass mask only when partially missing
        all_masked = padding_mask.all(dim=-1, keepdim=True)
        safe_mask = padding_mask.masked_fill(all_masked, False)

        attn_out, _ = self.mha(tokens, tokens, tokens, key_padding_mask=safe_mask)
        tokens = self.norm1(tokens + attn_out)

        # Flatten and project to (B, D)
        flat_tokens = tokens.flatten(start_dim=1)  # (B, num_planes * D)
        fused = self.norm2(self.proj(flat_tokens))
        return fused


class MultiPlaneKneeModel(nn.Module):
    """End-to-end 2.5D multi-plane vision model with MIL pooling and cross-plane fusion."""

    def __init__(
        self,
        backbone_name: str = "resnet34d",
        pretrained: bool = True,
        num_classes: int = len(DEFAULT_TARGET_COLS),
        planes: Tuple[str, ...] = PLANES_DEFAULT,
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize MultiPlaneKneeModel.

        Args:
            backbone_name: Architecture name from timm.
            pretrained: Whether to load pretrained vision weights.
            num_classes: Number of target abnormality outputs.
            planes: Tuple of anatomical planes to fuse.
            fusion_heads: Number of attention heads in MultiPlaneFusion.
            dropout: Dropout probability.
        """
        super().__init__()
        self.planes = planes
        self.num_classes = num_classes

        # Shared 2D slice feature extractor
        self.backbone = SliceBackbone(
            model_name=backbone_name, pretrained=pretrained
        )
        self.feature_dim = self.backbone.feature_dim

        # Dedicated MIL attention pool per plane
        self.pools = nn.ModuleDict(
            {p: GatedAttentionPool(self.feature_dim) for p in planes}
        )

        # Multi-plane cross-attention fusion layer
        self.fusion = MultiPlaneFusion(
            feature_dim=self.feature_dim,
            planes=planes,
            num_heads=fusion_heads,
            dropout=dropout,
        )

        # Final 12-class linear classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(
        self,
        plane_dict: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass over multi-plane slice tensors.

        Args:
            plane_dict: Dict mapping plane names ('sagittal', 'coronal', 'axial')
                        to (B, S, 3, H, W) slice tensors.
            return_attention: If True, also returns attention weights per plane.

        Returns:
            Logits tensor of shape (B, num_classes), or (logits, attention_dict).
        """
        plane_embeddings: Dict[str, torch.Tensor] = {}
        attention_weights: Dict[str, torch.Tensor] = {}

        for p_name in self.planes:
            if p_name in plane_dict and plane_dict[p_name] is not None:
                x = plane_dict[p_name]
                # Extract sequence slice features: (B, S, D)
                slice_feats = self.backbone(x)
                # Aggregate slices via MIL pooling: (B, D)
                pooled_feat, weights = self.pools[p_name](slice_feats)
                plane_embeddings[p_name] = pooled_feat
                attention_weights[p_name] = weights

        # Fuse plane embeddings
        fused_study_feat = self.fusion(plane_embeddings)  # (B, D)

        # Classification logits
        logits = self.classifier(fused_study_feat)  # (B, num_classes)

        if return_attention:
            return logits, attention_weights
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        plane_dict: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Compute continuous calibrated prediction probabilities in [0.0, 1.0].

        Args:
            plane_dict: Dict mapping plane names to slice sequence tensors.

        Returns:
            2D numpy array of shape (B, num_classes) with probabilities in [0.0, 1.0].
        """
        self.eval()
        logits = self.forward(plane_dict, return_attention=False)
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()
