"""Training package for RSNA Knee Abnormality Detection.

Provides class-balanced loss weighting, mixed-precision AMP trainer,
and multi-fold cross-validation experiment orchestration.
"""

from src.training.cv_experiment import CrossValidationExperiment
from src.training.loss import compute_pos_weights, create_loss_fn
from src.training.trainer import Trainer

__all__ = [
    "compute_pos_weights",
    "create_loss_fn",
    "Trainer",
    "CrossValidationExperiment",
]
