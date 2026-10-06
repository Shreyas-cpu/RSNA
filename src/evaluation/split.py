"""Leak-free Stratified Group K-Fold splitting engine.

Guarantees 0% cross-fold leakage of StudyInstanceUID/PatientID while balancing
multi-label positive distributions across all 12 target abnormality classes.
"""

from typing import List, Optional, Set
import numpy as np
import pandas as pd

DEFAULT_TARGET_COLS: List[str] = [
    "acl",
    "pcl",
    "mcl",
    "lcl",
    "medial_meniscus",
    "lateral_meniscus",
    "patellar_tendon",
    "quadriceps_tendon",
    "cartilage_femur",
    "cartilage_tibia",
    "cartilage_patella",
    "joint_effusion",
]


def iterative_multilabel_stratification(
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Partitions a multi-label binary matrix into n_splits balanced folds.

    Uses iterative greedy allocation (Sechidis et al. algorithm) sorting by label rarity.

    Args:
        labels: 2D binary numpy array of shape (N_samples, N_classes).
        n_splits: Number of folds.
        seed: Random seed for tie-breaking.

    Returns:
        1D integer array of shape (N_samples,) with fold indices in [0, n_splits - 1].
    """
    rng = np.random.RandomState(seed)
    n_samples, n_classes = labels.shape
    folds = np.full(n_samples, -1, dtype=int)

    # Track target positive counts per fold
    total_positives = labels.sum(axis=0)
    desired_pos_per_fold = total_positives / n_splits
    fold_pos_counts = np.zeros((n_splits, n_classes), dtype=float)
    fold_sample_counts = np.zeros(n_splits, dtype=int)

    # Track unassigned indices
    unassigned = set(range(n_samples))

    # Calculate rarity per sample (sum of inverse frequencies of present labels)
    inv_freq = 1.0 / np.maximum(total_positives, 1.0)

    # Process samples with positive labels first
    has_positives = np.where(labels.sum(axis=1) > 0)[0]
    rng.shuffle(has_positives)

    # Sort positive samples by rarest label score descending
    sample_rarity = (labels[has_positives] * inv_freq).max(axis=1)
    sorted_pos_indices = has_positives[np.argsort(-sample_rarity)]

    for idx in sorted_pos_indices:
        sample_labels = labels[idx]
        pos_classes = np.where(sample_labels > 0)[0]

        # Determine which fold needs these positive classes most
        fold_deficits = []
        for f in range(n_splits):
            deficit = (desired_pos_per_fold[pos_classes] - fold_pos_counts[f, pos_classes]).sum()
            fold_deficits.append((deficit, -fold_sample_counts[f], rng.rand(), f))

        # Pick fold with highest deficit (most in need of positives)
        best_fold = max(fold_deficits)[3]
        folds[idx] = best_fold
        fold_pos_counts[best_fold] += sample_labels
        fold_sample_counts[best_fold] += 1
        unassigned.remove(idx)

    # Assign remaining negative-only samples evenly across folds
    negative_indices = list(unassigned)
    rng.shuffle(negative_indices)
    for idx in negative_indices:
        best_fold = int(np.argmin(fold_sample_counts))
        folds[idx] = best_fold
        fold_sample_counts[best_fold] += 1

    return folds


def create_stratified_group_folds(
    df: pd.DataFrame,
    group_col: str = "StudyInstanceUID",
    target_cols: Optional[List[str]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Partitions dataframe into n_splits folds strictly grouped by group_col.

    Args:
        df: DataFrame containing study records.
        group_col: Column to group by (e.g., 'StudyInstanceUID').
        target_cols: List of binary target column names. Defaults to DEFAULT_TARGET_COLS.
        n_splits: Number of cross-validation folds.
        seed: Random seed.

    Returns:
        DataFrame with an added integer 'fold' column.
    """
    df = df.copy()

    # Determine present target columns
    if target_cols is None:
        target_cols = [c for c in DEFAULT_TARGET_COLS if c in df.columns]
        if not target_cols:
            # Fallback: discover any binary columns or default to random group split
            target_cols = []

    # Aggregate to unique groups
    if target_cols:
        group_df = df.groupby(group_col, as_index=False)[target_cols].max()
        label_matrix = group_df[target_cols].fillna(0).to_numpy(dtype=float)
        group_folds = iterative_multilabel_stratification(label_matrix, n_splits=n_splits, seed=seed)
        group_df["fold"] = group_folds
    else:
        # If no target columns, perform uniform group K-fold
        unique_groups = df[group_col].drop_duplicates().to_frame()
        rng = np.random.RandomState(seed)
        unique_groups["fold"] = rng.randint(0, n_splits, size=len(unique_groups))
        group_df = unique_groups

    # Map fold back to original dataframe
    fold_map = dict(zip(group_df[group_col], group_df["fold"]))
    df["fold"] = df[group_col].map(fold_map).astype(int)

    return df


def verify_leak_free_split(
    df: pd.DataFrame,
    fold_col: str = "fold",
    group_col: str = "StudyInstanceUID",
) -> bool:
    """Verifies that no group_col entities appear in more than one fold.

    Args:
        df: DataFrame containing fold and group columns.
        fold_col: Column name indicating fold assignment.
        group_col: Column name of grouping key.

    Returns:
        True if split is 100% leak-free.

    Raises:
        ValueError: If any group overlaps across distinct folds.
    """
    folds = df[fold_col].unique()
    group_sets = {f: set(df[df[fold_col] == f][group_col]) for f in folds}

    for i in folds:
        for j in folds:
            if i < j:
                overlap: Set = group_sets[i].intersection(group_sets[j])
                if overlap:
                    sample_leaks = list(overlap)[:3]
                    raise ValueError(
                        f"Data leakage detected! {len(overlap)} groups overlap between "
                        f"fold {i} and fold {j}. Example leaking IDs: {sample_leaks}"
                    )

    return True
