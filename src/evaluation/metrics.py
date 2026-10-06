"""Exact competition Macro ROC AUC evaluator and per-class diagnostic engine.

Calculates the unweighted arithmetic mean of ROC AUC across all 12 knee abnormality
classes using continuous prediction probabilities, with support for batch accumulation.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from src.evaluation.split import DEFAULT_TARGET_COLS


def compute_multilabel_auc_dict(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute individual ROC AUC scores for each target class.

    Args:
        y_true: Ground truth binary indicators of shape (N, C).
        y_pred: Continuous predicted probabilities in [0.0, 1.0] of shape (N, C).
        class_names: List of class names. Defaults to DEFAULT_TARGET_COLS.

    Returns:
        Dictionary mapping class name to ROC AUC float (or NaN if only 1 class present).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    n_samples, n_classes = y_true.shape
    if class_names is None:
        if n_classes == len(DEFAULT_TARGET_COLS):
            class_names = DEFAULT_TARGET_COLS
        else:
            class_names = [f"class_{i}" for i in range(n_classes)]

    scores: Dict[str, float] = {}
    for i in range(n_classes):
        c_name = class_names[i] if i < len(class_names) else f"class_{i}"
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # Check for valid binary distribution (must contain both 0 and 1)
        unique_vals = np.unique(col_true[~np.isnan(col_true)])
        if len(unique_vals) < 2:
            scores[c_name] = float("nan")
            continue

        try:
            auc = float(roc_auc_score(col_true, col_pred))
            scores[c_name] = auc
        except ValueError:
            scores[c_name] = float("nan")

    return scores


def compute_macro_auc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> float:
    """Compute exact unweighted Macro ROC AUC across all valid classes.

    Args:
        y_true: Ground truth binary indicators of shape (N, C).
        y_pred: Continuous predicted probabilities of shape (N, C).
        class_names: List of class names.

    Returns:
        Unweighted arithmetic mean of valid class AUCs (or 0.5 if none valid).
    """
    auc_dict = compute_multilabel_auc_dict(y_true, y_pred, class_names=class_names)
    valid_aucs = [v for v in auc_dict.values() if not np.isnan(v)]

    if not valid_aucs:
        return 0.5

    return float(np.mean(valid_aucs))


class MetricTracker:
    """Accumulates batch predictions during validation loops and generates metric reports."""

    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize MetricTracker with optional class names."""
        self.class_names = class_names or DEFAULT_TARGET_COLS
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated predictions."""
        self._y_true_batches: List[np.ndarray] = []
        self._y_pred_batches: List[np.ndarray] = []

    def update(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Accumulate a batch of ground truth and predictions.

        Args:
            y_true: Binary ground truth batch.
            y_pred: Predicted probability batch.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        self._y_true_batches.append(np.asarray(y_true, dtype=float))
        self._y_pred_batches.append(np.asarray(y_pred, dtype=float))

    def compute(self) -> Dict[str, Any]:
        """Compute macro and per-class AUC across all accumulated batches.

        Returns:
            Dictionary with 'macro_auc', 'per_class', 'support', and 'num_samples'.
        """
        if not self._y_true_batches:
            return {"macro_auc": 0.5, "per_class": {}, "num_samples": 0}

        y_true = np.concatenate(self._y_true_batches, axis=0)
        y_pred = np.concatenate(self._y_pred_batches, axis=0)

        per_class = compute_multilabel_auc_dict(y_true, y_pred, self.class_names)
        macro_auc = compute_macro_auc(y_true, y_pred, self.class_names)

        # Calculate positive support count per class
        support: Dict[str, int] = {}
        for i, c_name in enumerate(self.class_names):
            if i < y_true.shape[1]:
                support[c_name] = int(np.sum(y_true[:, i] == 1))

        return {
            "macro_auc": macro_auc,
            "per_class": per_class,
            "support": support,
            "num_samples": len(y_true),
        }

    def generate_summary_report(self) -> str:
        """Format an evaluation summary table highlighting per-class AUC performance.

        Returns:
            Formatted ASCII table string.
        """
        results = self.compute()
        macro_auc = results["macro_auc"]
        per_class = results["per_class"]
        support = results["support"]
        n_samples = results["num_samples"]

        lines = [
            "=" * 70,
            f"  VALIDATION EVALUATION REPORT (N = {n_samples})",
            f"  Macro-Averaged ROC AUC: {macro_auc:.4f}",
            "=" * 70,
            f"{'Pathology Class':<28} | {'Positives':<9} | {'Prevalence':<10} | {'AUC':<7} | {'Status'}",
            "-" * 70,
        ]

        for c_name in self.class_names:
            auc = per_class.get(c_name, float("nan"))
            pos = support.get(c_name, 0)
            prev = (pos / max(n_samples, 1)) * 100.0
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else " N/A "

            if np.isnan(auc):
                status = "⚪ NO_POS"
            elif auc >= 0.70:
                status = "🟢 EXCELLENT"
            elif auc >= 0.584:
                status = "🟢 BEATS_V9"
            elif auc >= 0.55:
                status = "🟡 PROGRESS"
            else:
                status = "🔴 WEAK (<0.55)"

            lines.append(f"{c_name:<28} | {pos:<9} | {prev:>8.1f}% | {auc_str:<7} | {status}")

        lines.append("=" * 70)
        return "\n".join(lines)
