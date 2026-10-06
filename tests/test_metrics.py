import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    MetricTracker,
    compute_macro_auc,
    compute_multilabel_auc_dict,
)
from src.evaluation.split import DEFAULT_TARGET_COLS


def test_compute_macro_auc_perfect():
    """Verify perfect predictions yield 1.0 Macro AUC across all classes."""
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ])
    y_pred = y_true.astype(float)

    auc_dict = compute_multilabel_auc_dict(y_true, y_pred, class_names=["c1", "c2", "c3"])
    macro = compute_macro_auc(y_true, y_pred, class_names=["c1", "c2", "c3"])

    assert auc_dict["c1"] == 1.0
    assert auc_dict["c2"] == 1.0
    assert auc_dict["c3"] == 1.0
    assert macro == 1.0


def test_compute_macro_auc_inverted():
    """Verify perfectly inverse predictions yield 0.0 Macro AUC."""
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ])
    y_pred = 1.0 - y_true.astype(float)

    macro = compute_macro_auc(y_true, y_pred, class_names=["c1", "c2"])
    assert macro == 0.0


def test_compute_macro_auc_random():
    """Verify uninformative random predictions yield ~0.5 Macro AUC."""
    np.random.seed(42)
    n = 2000
    y_true = np.random.binomial(1, 0.3, size=(n, 4))
    y_pred = np.random.uniform(0.0, 1.0, size=(n, 4))

    macro = compute_macro_auc(y_true, y_pred)
    assert pytest.approx(macro, abs=0.05) == 0.5


def test_compute_macro_auc_single_class_edge_case():
    """Verify single-class presence returns NaN for that class without crashing macro calculation."""
    y_true = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 1],  # Class 0 has only 1s! Class 1 has both 0 and 1.
    ])
    y_pred = np.array([
        [0.8, 0.1],
        [0.7, 0.2],
        [0.9, 0.3],
        [0.6, 0.9],
    ])

    auc_dict = compute_multilabel_auc_dict(y_true, y_pred, class_names=["only_ones", "valid"])
    assert np.isnan(auc_dict["only_ones"])
    assert auc_dict["valid"] == 1.0

    # Macro AUC should average only over valid classes -> 1.0
    macro = compute_macro_auc(y_true, y_pred, class_names=["only_ones", "valid"])
    assert macro == 1.0


def test_metric_tracker_batch_accumulation():
    """Verify MetricTracker accumulates PyTorch tensor batches and computes correct summary."""
    tracker = MetricTracker(class_names=DEFAULT_TARGET_COLS)

    # Batch 1
    t1 = torch.tensor([[1, 0] + [0] * 10, [0, 1] + [0] * 10], dtype=torch.float32)
    p1 = torch.tensor([[0.9, 0.1] + [0.5] * 10, [0.2, 0.8] + [0.5] * 10], dtype=torch.float32)
    tracker.update(t1, p1)

    # Batch 2
    t2 = torch.tensor([[1, 1] + [0] * 10, [0, 0] + [0] * 10], dtype=torch.float32)
    p2 = torch.tensor([[0.8, 0.7] + [0.5] * 10, [0.1, 0.2] + [0.5] * 10], dtype=torch.float32)
    tracker.update(t2, p2)

    res = tracker.compute()
    assert res["num_samples"] == 4
    assert res["per_class"]["acl"] == 1.0
    assert res["per_class"]["pcl"] == 1.0

    report = tracker.generate_summary_report()
    assert "VALIDATION EVALUATION REPORT" in report
    assert "Macro-Averaged ROC AUC" in report
    assert "acl" in report


def test_end_to_end_cross_validation_harness():
    """Verify end-to-end 5-fold CV simulation: splitting, tracking, per-fold and OOF evaluation."""
    import pandas as pd
    from src.evaluation.split import create_stratified_group_folds, verify_leak_free_split

    rng = np.random.RandomState(42)
    n_studies = 150
    rows = []

    # 1. Create multi-study dataset
    for i in range(n_studies):
        sid = f"study_{i:04d}"
        labels = {c: rng.binomial(1, 0.25) for c in DEFAULT_TARGET_COLS}
        # Multi-series representation
        for s_idx in range(rng.randint(1, 3)):
            rows.append({"StudyInstanceUID": sid, "Series": f"s_{s_idx}", **labels})

    df = pd.DataFrame(rows)

    # 2. Split into 5 folds
    split_df = create_stratified_group_folds(df, group_col="StudyInstanceUID", n_splits=5, seed=42)
    assert verify_leak_free_split(split_df) is True

    # 3. Simulate 5-fold cross-validation
    oof_preds = np.zeros((len(split_df), len(DEFAULT_TARGET_COLS)))
    oof_trues = split_df[DEFAULT_TARGET_COLS].to_numpy(dtype=float)
    global_tracker = MetricTracker(class_names=DEFAULT_TARGET_COLS)

    fold_macro_aucs = []

    for fold in range(5):
        val_mask = (split_df["fold"] == fold).to_numpy()
        val_y = oof_trues[val_mask]

        # Simulate model predicting signal + noise:
        # Signal correlates with true label (simulating ~0.75 AUC model)
        val_pred = np.clip(0.5 * val_y + 0.3 * rng.uniform(0.0, 1.0, size=val_y.shape), 0.0, 1.0)
        oof_preds[val_mask] = val_pred

        fold_tracker = MetricTracker(class_names=DEFAULT_TARGET_COLS)
        fold_tracker.update(val_y, val_pred)
        fold_res = fold_tracker.compute()

        fold_macro_aucs.append(fold_res["macro_auc"])
        global_tracker.update(val_y, val_pred)

    # 4. Verify all folds executed and beat random guess (0.5)
    assert len(fold_macro_aucs) == 5
    for f_auc in fold_macro_aucs:
        assert f_auc > 0.65, f"Fold AUC too low: {f_auc}"

    # 5. Global OOF evaluation
    global_results = global_tracker.compute()
    oof_macro_auc = global_results["macro_auc"]
    assert oof_macro_auc > 0.65

    # 6. Verify summary report generation
    report = global_tracker.generate_summary_report()
    assert "VALIDATION EVALUATION REPORT" in report
    assert f"{oof_macro_auc:.4f}" in report
    print("\n" + report)
