"""Kaggle Submission Schema Validator & Model Efficiency Benchmark Profiler.

Validates submission.csv formatting, null-guards, value bounds in [0.0, 1.0],
and benchmarks inference speed to certify qualification for the $17,000 Efficiency Track.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.split import DEFAULT_TARGET_COLS


class SubmissionValidator:
    """Certifies submission CSV compliance and benchmarks inference latency."""

    def __init__(self, target_cols: Optional[List[str]] = None):
        """Initialize SubmissionValidator.

        Args:
            target_cols: List of target column names in exact expected order.
        """
        self.target_cols = target_cols or DEFAULT_TARGET_COLS

    def validate_submission_df(
        self,
        sub_df: pd.DataFrame,
        expected_test_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate an in-memory submission DataFrame.

        Args:
            sub_df: DataFrame to validate.
            expected_test_ids: Optional list of expected StudyInstanceUID values.

        Returns:
            Dict containing validation summary.

        Raises:
            ValueError: If schema, values, or ordering violate Kaggle requirements.
        """
        if sub_df is None or not isinstance(sub_df, pd.DataFrame):
            raise ValueError("Input submission is not a valid pandas DataFrame.")

        # 1. Column presence check
        if "StudyInstanceUID" not in sub_df.columns:
            raise ValueError("Missing required identifier column 'StudyInstanceUID'.")

        missing_targets = [col for col in self.target_cols if col not in sub_df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        expected_total_cols = len(self.target_cols) + 1
        if len(sub_df.columns) != expected_total_cols:
            extra_cols = [c for c in sub_df.columns if c not in self.target_cols and c != "StudyInstanceUID"]
            raise ValueError(f"Found unexpected extra columns in submission: {extra_cols}")

        # 2. Row count and ID ordering
        if len(sub_df) == 0:
            raise ValueError("Submission file is empty (0 rows).")

        if expected_test_ids is not None:
            if len(sub_df) != len(expected_test_ids):
                raise ValueError(
                    f"Row count mismatch: expected {len(expected_test_ids)}, found {len(sub_df)}."
                )
            if list(sub_df["StudyInstanceUID"]) != list(expected_test_ids):
                raise ValueError("StudyInstanceUID ordering or values do not match expected test set.")

        # 3. Null / NaN / Inf checks
        target_vals = sub_df[self.target_cols].to_numpy(dtype=float)
        if np.isnan(target_vals).any():
            nan_count = int(np.isnan(target_vals).sum())
            raise ValueError(f"Submission contains {nan_count} NaN values.")

        if np.isinf(target_vals).any():
            inf_count = int(np.isinf(target_vals).sum())
            raise ValueError(f"Submission contains {inf_count} infinite values.")

        # 4. Probability range bounds [0.0, 1.0]
        min_val = float(np.min(target_vals))
        max_val = float(np.max(target_vals))
        if min_val < 0.0 or max_val > 1.0:
            raise ValueError(
                f"Probabilities out of range: min={min_val:.4f}, max={max_val:.4f}. Must be in [0.0, 1.0]."
            )

        return {
            "valid": True,
            "n_rows": len(sub_df),
            "n_columns": len(sub_df.columns),
            "columns": list(sub_df.columns),
            "min_prob": min_val,
            "max_prob": max_val,
        }

    def validate_submission_file(
        self,
        sub_csv_path: Union[str, Path],
        test_csv_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Validate submission CSV file from disk."""
        sub_file = Path(sub_csv_path)
        if not sub_file.exists():
            raise FileNotFoundError(f"Submission file does not exist: {sub_file}")

        sub_df = pd.read_csv(sub_file)

        expected_ids = None
        if test_csv_path is not None and Path(test_csv_path).exists():
            test_df = pd.read_csv(test_csv_path)
            if "StudyInstanceUID" in test_df.columns:
                expected_ids = list(test_df["StudyInstanceUID"])

        return self.validate_submission_df(sub_df, expected_test_ids=expected_ids)

    def benchmark_inference_efficiency(
        self,
        model: nn.Module,
        n_studies: int = 10,
        target_minutes: float = 30.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Profile model inference latency and project total test set runtime.

        Args:
            model: PyTorch vision model.
            n_studies: Number of simulated benchmark studies to process.
            target_minutes: Upper runtime bound for efficiency track (30.0 min).
            device: Computing device.

        Returns:
            Dict containing latency stats and efficiency prize qualification status.
        """
        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        model = model.to(dev)
        model.eval()

        # Warmup
        warmup_planes = {
            "sagittal": torch.randn(1, 4, 3, 256, 256, device=dev),
            "coronal": torch.randn(1, 4, 3, 256, 256, device=dev),
            "axial": torch.randn(1, 4, 3, 256, 256, device=dev),
        }
        with torch.inference_mode():
            for _ in range(2):
                _ = model(warmup_planes)

        # Benchmark run
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_studies):
                # Simulate 4 slabs per plane
                planes = {
                    "sagittal": torch.randn(1, 4, 3, 256, 256, device=dev),
                    "coronal": torch.randn(1, 4, 3, 256, 256, device=dev),
                    "axial": torch.randn(1, 4, 3, 256, 256, device=dev),
                }
                _ = model(planes)

        elapsed = time.perf_counter() - start_time
        sec_per_study = elapsed / n_studies
        ms_per_study = sec_per_study * 1000.0
        studies_per_sec = 1.0 / max(sec_per_study, 1e-6)

        # Extrapolate for 1000 and 2000 test cases
        proj_min_1000 = (sec_per_study * 1000.0) / 60.0
        proj_min_2000 = (sec_per_study * 2000.0) / 60.0

        qualifies = proj_min_2000 < target_minutes

        return {
            "n_studies_benchmarked": n_studies,
            "device": str(dev),
            "total_elapsed_sec": elapsed,
            "ms_per_study": ms_per_study,
            "studies_per_second": studies_per_sec,
            "projected_minutes_1000": proj_min_1000,
            "projected_minutes_2000": proj_min_2000,
            "target_minutes": target_minutes,
            "qualifies_for_efficiency_track": qualifies,
        }
