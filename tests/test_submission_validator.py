"""Unit and integration tests for SubmissionValidator and efficiency benchmark."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.models.multi_plane import MultiPlaneKneeModel
from src.submission.validator import SubmissionValidator


@pytest.fixture
def validator():
    return SubmissionValidator()


def create_valid_dummy_submission(n_rows: int = 5) -> pd.DataFrame:
    """Helper creating a compliant submission DataFrame."""
    records = []
    for i in range(n_rows):
        rec = {"StudyInstanceUID": f"test_study_{i:03d}"}
        for col in DEFAULT_TARGET_COLS:
            rec[col] = float((i + 1) * 0.05)
        records.append(rec)
    return pd.DataFrame(records)


def test_validate_submission_df_valid(validator):
    """Verify clean submission DataFrame passes validation with complete summary."""
    df = create_valid_dummy_submission(n_rows=4)
    summary = validator.validate_submission_df(df)

    assert summary["valid"] is True
    assert summary["n_rows"] == 4
    assert summary["n_columns"] == 13
    assert 0.0 <= summary["min_prob"] <= 1.0
    assert 0.0 <= summary["max_prob"] <= 1.0


def test_validate_submission_df_detects_nans(validator):
    """Verify validation fails when NaN values are present."""
    df = create_valid_dummy_submission(n_rows=4)
    df.loc[1, "acl"] = np.nan

    with pytest.raises(ValueError, match="NaN values"):
        validator.validate_submission_df(df)


def test_validate_submission_df_detects_out_of_bounds(validator):
    """Verify validation fails when probabilities exceed [0.0, 1.0]."""
    df = create_valid_dummy_submission(n_rows=4)
    df.loc[2, "joint_effusion"] = 1.05

    with pytest.raises(ValueError, match="out of range"):
        validator.validate_submission_df(df)

    df.loc[2, "joint_effusion"] = -0.01
    with pytest.raises(ValueError, match="out of range"):
        validator.validate_submission_df(df)


def test_validate_submission_df_detects_missing_cols(validator):
    """Verify validation catches missing columns."""
    df = create_valid_dummy_submission(n_rows=4)
    df = df.drop(columns=["acl"])

    with pytest.raises(ValueError, match="Missing target columns"):
        validator.validate_submission_df(df)


def test_validate_submission_file_and_ids(validator, tmp_path):
    """Verify file-based validation with expected test set identifiers."""
    df = create_valid_dummy_submission(n_rows=3)
    sub_path = tmp_path / "submission.csv"
    df.to_csv(sub_path, index=False)

    test_df = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"]})
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    summary = validator.validate_submission_file(sub_path, test_csv_path=test_path)
    assert summary["valid"] is True
    assert summary["n_rows"] == 3


def test_benchmark_inference_efficiency(validator):
    """Verify efficiency benchmark profiles throughput and certifies < 30m constraint."""
    model = MultiPlaneKneeModel(
        backbone_name="resnet34d",
        pretrained=False,
        fusion_heads=2,
    )

    stats = validator.benchmark_inference_efficiency(
        model=model,
        n_studies=2,
        target_minutes=30.0,
    )

    assert "ms_per_study" in stats
    assert stats["ms_per_study"] > 0
    assert stats["qualifies_for_efficiency_track"] is True
    assert stats["projected_minutes_2000"] < 30.0
