"""Class-balanced multi-label loss engine for RSNA Knee Abnormality Detection.

Calculates square-root dampened positive class weights to mitigate severe
label imbalance (rare pathologies ~3% vs common pathologies ~30%) and creates
configured PyTorch BCEWithLogitsLoss instances.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.split import DEFAULT_TARGET_COLS


def compute_pos_weights(
    df_or_array: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    target_cols: Optional[List[str]] = None,
    clip_range: Tuple[float, float] = (1.0, 10.0),
) -> torch.Tensor:
    """Compute square-root dampened positive class weights for multi-label BCE.

    Formula: w_c = clip(sqrt(N_neg / N_pos), min_w, max_w)

    Args:
        df_or_array: DataFrame, numpy array, or tensor containing binary ground truth.
        target_cols: List of target column names if DataFrame provided.
        clip_range: Tuple (min_weight, max_weight) to bound scaling factors.

    Returns:
        1D torch.Tensor of shape (num_classes,) with dtype float32.
    """
    if isinstance(df_or_array, pd.DataFrame):
        cols = target_cols or [c for c in DEFAULT_TARGET_COLS if c in df_or_array.columns]
        if not cols:
            raise ValueError("No target columns found in DataFrame")
        labels = df_or_array[cols].fillna(0).to_numpy(dtype=float)
    elif isinstance(df_or_array, torch.Tensor):
        labels = df_or_array.detach().cpu().numpy().astype(float)
    else:
        labels = np.asarray(df_or_array, dtype=float)

    if labels.ndim == 1:
        labels = labels[:, np.newaxis]

    n_samples, n_classes = labels.shape
    weights = []
    min_w, max_w = clip_range

    for c in range(n_classes):
        col = labels[:, c]
        n_pos = float(np.sum(col == 1))
        n_neg = float(np.sum(col == 0))

        if n_pos > 0:
            raw_w = np.sqrt(n_neg / n_pos)
        else:
            raw_w = max_w

        clipped_w = float(np.clip(raw_w, min_w, max_w))
        weights.append(clipped_w)

    return torch.tensor(weights, dtype=torch.float32)


def create_loss_fn(
    pos_weights: Optional[torch.Tensor] = None,
) -> nn.BCEWithLogitsLoss:
    """Create a BCEWithLogitsLoss function configured with positive class weights.

    Args:
        pos_weights: Optional 1D tensor of shape (num_classes,).

    Returns:
        nn.BCEWithLogitsLoss instance.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
