"""Weak Supervision Label Validator and Silver Dataset Generator.

Evaluates NLP pseudo-labels against ground-truth labels and generates certainty-weighted
silver training datasets for vision model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.nlp.report_parser import ClinicalReportParser


class WeakLabelValidator:
    """Validates weak labels against ground truth and generates silver datasets."""

    def __init__(
        self,
        parser: Optional[ClinicalReportParser] = None,
        target_cols: Optional[List[str]] = None,
    ):
        """Initialize WeakLabelValidator.

        Args:
            parser: Optional ClinicalReportParser instance.
            target_cols: Target column names.
        """
        self.target_cols = target_cols or DEFAULT_TARGET_COLS
        self.parser = parser or ClinicalReportParser(target_cols=self.target_cols)

    def evaluate_alignment(
        self,
        df: pd.DataFrame,
        report_col: str = "Report",
        target_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute alignment metrics (Macro AUC, Pearson correlation, F1) vs ground truth.

        Args:
            df: DataFrame containing report column and ground-truth binary columns.
            report_col: Name of the report column.
            target_cols: List of target columns to evaluate.

        Returns:
            Dict containing macro_auc, mean_correlation, mean_f1, and per-class breakdown.
        """
        targets = target_cols or self.target_cols

        # Filter rows having valid report and non-null ground-truth values
        valid_mask = df[report_col].notna() & (df[report_col].str.strip() != "")
        for t in targets:
            if t in df.columns:
                valid_mask = valid_mask & df[t].notna()

        subset = df[valid_mask].copy()
        if len(subset) == 0:
            raise ValueError("No valid rows with both reports and ground-truth labels found.")

        # Extract pseudo-labels
        pred_df = self.parser.parse_dataframe(subset, report_col=report_col)

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        aucs: List[float] = []
        corrs: List[float] = []
        f1s: List[float] = []

        for col in targets:
            y_true = subset[col].values.astype(float)
            y_pred = pred_df[col].values.astype(float)

            # Compute ROC AUC if both 0 and 1 are present
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                try:
                    auc = float(roc_auc_score(y_true, y_pred))
                except Exception:
                    auc = 0.5
            else:
                auc = 1.0 if np.all((y_pred >= 0.5) == (y_true == 1)) else 0.5

            # Compute Pearson correlation
            if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
                corr = float(np.corrcoef(y_true, y_pred)[0, 1])
            else:
                corr = 0.0

            # Compute F1 at 0.5 threshold
            binary_pred = (y_pred >= 0.5).astype(int)
            binary_true = (y_true >= 0.5).astype(int)
            f1 = float(f1_score(binary_true, binary_pred, zero_division=1.0))

            per_class_metrics[col] = {
                "auc": auc,
                "correlation": corr,
                "f1": f1,
                "pos_rate_true": float(np.mean(y_true)),
                "pos_rate_pred": float(np.mean(binary_pred)),
            }
            aucs.append(auc)
            corrs.append(corr)
            f1s.append(f1)

        macro_auc = float(np.mean(aucs)) if aucs else 0.5
        mean_corr = float(np.mean(corrs)) if corrs else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0

        return {
            "macro_auc": macro_auc,
            "mean_correlation": mean_corr,
            "mean_f1": mean_f1,
            "n_samples": len(subset),
            "per_class": per_class_metrics,
        }

    def generate_silver_dataset(
        self,
        df: pd.DataFrame,
        report_col: str = "Report",
        target_cols: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        min_certainty: float = 0.1,
    ) -> pd.DataFrame:
        """Extract pseudo-labels, compute sample weights, and generate silver dataset.

        Args:
            df: Input DataFrame containing reports.
            report_col: Name of the report column.
            target_cols: Target column names.
            output_path: Optional file path to export CSV.
            min_certainty: Minimum sample certainty threshold for inclusion.

        Returns:
            DataFrame containing study identifiers, continuous pseudo-labels, and sample_weight.
        """
        targets = target_cols or self.target_cols
        pred_df = self.parser.parse_dataframe(df, report_col=report_col)

        # Calculate sample certainty weight: w_i = mean_c(2 * |p_ic - 0.5|)
        # Range is strictly [0.0, 1.0]
        prob_matrix = pred_df[targets].values
        certainties = np.mean(2.0 * np.abs(prob_matrix - 0.5), axis=1)

        result_df = pd.DataFrame(index=df.index)
        if "StudyInstanceUID" in df.columns:
            result_df["StudyInstanceUID"] = df["StudyInstanceUID"]

        for col in targets:
            result_df[col] = pred_df[col]

        result_df["sample_weight"] = np.clip(certainties, 0.0, 1.0)

        # Filter by minimum certainty
        filtered_df = result_df[result_df["sample_weight"] >= min_certainty].copy()

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            filtered_df.to_csv(out_file, index=False)

        return filtered_df
