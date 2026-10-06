"""Unit and integration tests for WeakLabelValidator and silver dataset generator."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.nlp.validator import WeakLabelValidator


@pytest.fixture
def validator():
    return WeakLabelValidator()


def test_validator_alignment_metrics(validator):
    """Verify evaluate_alignment computes high Macro AUC and correlation on aligned data."""
    # Construct synthetic reports with known ground-truth labels
    records = []
    for i in range(20):
        is_acl_tear = int(i % 2 == 1)
        is_effusion = int(i % 3 == 0)

        report_lines = []
        if is_acl_tear:
            report_lines.append("Complete tear of anterior cruciate ligament (ACL).")
        else:
            report_lines.append("Anterior cruciate ligament is completely intact.")

        if is_effusion:
            report_lines.append("Moderate joint effusion present.")
        else:
            report_lines.append("No joint effusion.")

        # Add neutral findings for other structures
        report_lines.append("Intact menisci. Normal cartilage and tendons.")

        record = {
            "StudyInstanceUID": f"study_{i:03d}",
            "Report": " ".join(report_lines),
            "acl": float(is_acl_tear),
            "joint_effusion": float(is_effusion),
        }
        # Populate other targets with 0.0
        for col in DEFAULT_TARGET_COLS:
            if col not in record:
                record[col] = 0.0

        records.append(record)

    df = pd.DataFrame(records)
    metrics = validator.evaluate_alignment(df, report_col="Report")

    assert "macro_auc" in metrics
    assert "mean_correlation" in metrics
    assert "per_class" in metrics
    assert metrics["n_samples"] == 20

    # ACL should have near-perfect alignment (> 0.95 AUC)
    assert metrics["per_class"]["acl"]["auc"] >= 0.90
    assert metrics["per_class"]["acl"]["correlation"] >= 0.80

    # Overall macro AUC across evaluated targets should be high
    assert metrics["macro_auc"] >= 0.80


def test_validator_empty_raises_value_error(validator):
    """Verify ValueError is raised when no rows have both reports and labels."""
    df = pd.DataFrame({"StudyInstanceUID": ["s1"], "Report": [""]})
    with pytest.raises(ValueError):
        validator.evaluate_alignment(df, report_col="Report")


def test_generate_silver_dataset(validator, tmp_path):
    """Verify silver dataset generation, sample weighting, and CSV export."""
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s2", "s3"],
            "Report": [
                "Complete full-thickness ACL tear and large joint effusion.",
                "Normal knee MRI. Intact ligaments and menisci. No effusion.",
                "Equivocal subtle periligamentous signal.",
            ],
        }
    )

    out_csv = tmp_path / "silver_train.csv"
    silver_df = validator.generate_silver_dataset(
        df, report_col="Report", output_path=out_csv, min_certainty=0.0
    )

    assert out_csv.exists()
    assert len(silver_df) == 3
    assert "sample_weight" in silver_df.columns
    assert "StudyInstanceUID" in silver_df.columns

    for col in DEFAULT_TARGET_COLS:
        assert col in silver_df.columns
        assert silver_df[col].between(0.0, 1.0).all()

    # Clear reports (s1 and s2) should have higher sample weights than ambiguous (s3)
    assert silver_df.loc[0, "sample_weight"] > silver_df.loc[2, "sample_weight"]
    assert silver_df.loc[1, "sample_weight"] > silver_df.loc[2, "sample_weight"]
