import numpy as np
import pandas as pd
import pytest

from src.evaluation.split import (
    DEFAULT_TARGET_COLS,
    create_stratified_group_folds,
    verify_leak_free_split,
)


def test_create_stratified_group_folds_zero_leakage():
    """Verify create_stratified_group_folds creates exactly 5 folds with 0 study leakage."""
    rng = np.random.RandomState(42)
    n_studies = 100
    rows = []

    # Simulate studies having 1 to 3 series rows each
    for i in range(n_studies):
        sid = f"study_{i:04d}"
        n_series = rng.randint(1, 4)
        # Random multi-label binary targets
        labels = {c: rng.binomial(1, 0.2) for c in DEFAULT_TARGET_COLS}
        for s_idx in range(n_series):
            row = {"StudyInstanceUID": sid, "SeriesID": f"series_{i}_{s_idx}", **labels}
            rows.append(row)

    df = pd.DataFrame(rows)
    split_df = create_stratified_group_folds(df, group_col="StudyInstanceUID", n_splits=5, seed=42)

    assert "fold" in split_df.columns
    assert sorted(split_df["fold"].unique().tolist()) == [0, 1, 2, 3, 4]

    # Verify zero leakage
    assert verify_leak_free_split(split_df, fold_col="fold", group_col="StudyInstanceUID") is True

    # Verify every row with the same StudyInstanceUID has the exact same fold
    study_fold_counts = split_df.groupby("StudyInstanceUID")["fold"].nunique()
    assert (study_fold_counts == 1).all()


def test_verify_leak_free_split_detects_leakage():
    """Verify verify_leak_free_split raises ValueError when a study appears in multiple folds."""
    df = pd.DataFrame([
        {"StudyInstanceUID": "study_001", "fold": 0},
        {"StudyInstanceUID": "study_002", "fold": 1},
        {"StudyInstanceUID": "study_001", "fold": 2},  # Leaking into fold 2!
    ])

    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_leak_free_split(df, fold_col="fold", group_col="StudyInstanceUID")


def test_stratified_group_folds_rare_class_distribution():
    """Verify rare positive classes are distributed across all folds."""
    rng = np.random.RandomState(42)
    n_studies = 120
    rows = []

    for i in range(n_studies):
        sid = f"study_{i:04d}"
        # Set 'pcl' to very rare ~ 10% prevalence (12 positives)
        pcl_val = 1 if i < 12 else 0
        acl_val = 1 if i % 3 == 0 else 0
        row = {"StudyInstanceUID": sid, "pcl": pcl_val, "acl": acl_val}
        rows.append(row)

    df = pd.DataFrame(rows)
    split_df = create_stratified_group_folds(
        df,
        group_col="StudyInstanceUID",
        target_cols=["pcl", "acl"],
        n_splits=5,
        seed=42,
    )

    assert verify_leak_free_split(split_df) is True

    # Check that 'pcl' positives exist in every single fold
    fold_pcl_counts = split_df.groupby("fold")["pcl"].sum()
    for f in range(5):
        assert fold_pcl_counts[f] > 0, f"Fold {f} has 0 pcl positives!"
