"""Models package for RSNA Knee Abnormality Detection.

Provides 2.5D multi-slice vision feature extraction, Gated Attention pooling,
and 12-class multi-label classification.
"""

from src.models.attention import GatedAttentionPool
from src.models.backbone import SliceBackbone
from src.models.model import RSNAKneeModel

__all__ = [
    "SliceBackbone",
    "GatedAttentionPool",
    "RSNAKneeModel",
]
