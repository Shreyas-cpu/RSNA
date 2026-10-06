import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import RSNAKneeDataset
from src.evaluation.split import DEFAULT_TARGET_COLS
from src.training.cv_experiment import CrossValidationExperiment


def test_cross_validation_experiment_run(tmp_path):
    """Verify CrossValidationExperiment trains multi-fold CV and aggregates valid OOF predictions."""
    cache_dir = tmp_path / "cv_cache"
    ckpt_dir = tmp_path / "cv_checkpoints"

    n_studies = 6
    rows = []

    # Create synthetic studies with slabs cached
    for i in range(n_studies):
        sid = f"study_{i:03d}"
        labels = {c: float((i + j) % 2) for j, c in enumerate(DEFAULT_TARGET_COLS)}
        rows.append({"StudyInstanceUID": sid, **labels})

        # Cache synthetic 2.5D slabs: 2 slabs * 3 planes = 6 slabs total
        vols = {
            "sagittal": np.random.uniform(0, 1000, size=(4, 32, 32)).astype(np.float32),
            "coronal": np.random.uniform(0, 1000, size=(4, 32, 32)).astype(np.float32),
            "axial": np.random.uniform(0, 1000, size=(4, 32, 32)).astype(np.float32),
        }
        RSNAKneeDataset.cache_study_slabs(
            study_id=sid,
            series_volumes=vols,
            output_dir=cache_dir,
            num_slabs=2,
            target_shape=(32, 32),
        )

    df = pd.DataFrame(rows)

    # Initialize experiment with 2 splits and 1 epoch for rapid unit test
    exp = CrossValidationExperiment(
        df=df,
        cache_dir=cache_dir,
        target_cols=DEFAULT_TARGET_COLS,
        backbone_name="resnet34d",
        epochs=1,
        batch_size=2,
        lr=1e-4,
        n_splits=2,
        checkpoint_dir=ckpt_dir,
        use_amp=True,
    )

    # Test single fold execution
    val_preds, fold_auc = exp.run_fold(0)
    assert val_preds.shape[1] == len(DEFAULT_TARGET_COLS)
    assert (val_preds >= 0.0).all() and (val_preds <= 1.0).all()

    # Test all folds execution
    results = exp.run_all_folds()
    assert "oof_macro_auc" in results
    assert "fold_aucs" in results
    assert len(results["fold_aucs"]) == 2
    assert results["oof_predictions"].shape == (n_studies, len(DEFAULT_TARGET_COLS))

    # Checkpoint folders created
    assert (ckpt_dir / "fold_0" / "best_model.pt").exists()
    assert (ckpt_dir / "fold_1" / "best_model.pt").exists()

    # Formatted evaluation report exists
    assert "VALIDATION EVALUATION REPORT" in results["report"]
    print("\n" + results["report"])
