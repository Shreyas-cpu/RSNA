# Phase 2 Context: Leakage-Free Stratified Group Validation Harness

## Overview
Phase 2 builds the core validation infrastructure required to measure genuine model improvements and prevent catastrophic data leakage across patient exams. It implements multi-label Stratified Group K-Fold splitting and exact Macro ROC AUC evaluation across the 12 target abnormality classes.

## Decisions

- **D-07:** Leakage-Free Stratified Group K-Fold (EVAL-01) — Partition studies strictly by `StudyInstanceUID` with multi-label stratification to guarantee 0% cross-fold leakage while preserving rare pathology balance.
- **D-08:** Exact Macro ROC AUC Metric Evaluator (EVAL-02) — Calculate the unweighted arithmetic mean of ROC AUC across all 12 abnormality target classes using continuous predicted probabilities.
- **D-09:** Granular Per-Class Performance Diagnostic (EVAL-02) — Track and log per-class AUC breakdowns and support counts to prevent silent metric collapse on rare classes.

## Key Artifacts to Produce
- `src/evaluation/__init__.py`: Package init.
- `src/evaluation/metrics.py`: `compute_macro_auc`, `compute_multilabel_auc_dict`, `MetricTracker`.
- `src/evaluation/split.py`: `create_stratified_group_folds`, `verify_leak_free_split`.
- `tests/test_metrics.py`: Unit tests for Macro AUC calculation, perfect score, random score (0.5), inverse score, and NaN edge cases.
- `tests/test_split.py`: Unit tests verifying zero study leakage across folds and balanced multi-label class distribution.
