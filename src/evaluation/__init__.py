"""Evaluation package for RSNA Knee Abnormality Detection.

Provides leak-free stratified group splitting, exact competition Macro ROC AUC evaluation,
and granular per-pathology diagnostics.
"""

from src.evaluation.metrics import (
    MetricTracker,
    compute_macro_auc,
    compute_multilabel_auc_dict,
)
from src.evaluation.split import (
    DEFAULT_TARGET_COLS,
    create_stratified_group_folds,
    verify_leak_free_split,
)

__all__ = [
    "DEFAULT_TARGET_COLS",
    "create_stratified_group_folds",
    "verify_leak_free_split",
    "compute_multilabel_auc_dict",
    "compute_macro_auc",
    "MetricTracker",
]
