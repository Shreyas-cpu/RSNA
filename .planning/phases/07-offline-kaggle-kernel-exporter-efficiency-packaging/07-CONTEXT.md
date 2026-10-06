# Phase 7 Context: Offline Kaggle Kernel Exporter & Efficiency Packaging

## Overview
Phase 7 delivers the final submission vehicle for the RSNA Knee Abnormality Detection competition. In Kaggle code competitions, submission kernels run strictly in an **air-gapped, offline environment** with no internet access. Furthermore, this competition features a dedicated **$17,000 Model Efficiency Track** requiring submissions to finish complete test set inference in **< 30 minutes**. This phase builds the automated exporter that packages our high-speed DICOM reader, 2.5D slab extraction, multi-plane attention fusion model, and calibrated probabilities into a self-contained submission script/notebook, verifies strict Kaggle schema compliance, benchmarks runtime speed, and generates Kaggle CLI push metadata.

## Decisions

- **D-21:** Self-Contained Offline Exporter (SUBMIT-01, SUBMIT-02) — Build `SubmissionExporter` which generates a standalone, self-contained Python script (`submission.py`) and equivalent Jupyter Notebook (`rsna_knee_submission.ipynb`). The exporter embeds required modules or bundles wheel installation instructions, loads test series from `/kaggle/input/rsna-knee-abnormality-detection/`, and runs batched inference with torch.inference_mode() and fp16 AMP.
- **D-22:** Submission Validator & Efficiency Benchmark Gate (SUBMIT-03, SUBMIT-04) — Implement `SubmissionValidator` that verifies generated `submission.csv` files against the official Kaggle criteria: exact column schema, all continuous floats in $[0.0, 1.0]$, zero NaNs/nulls, and matching test `StudyInstanceUID` index. Profiles end-to-end inference throughput ensuring runtime qualifies for the $17,000 Efficiency Prize (< 30 minutes total test set inference). Generates `kernel-metadata.json` for one-command CLI deployment.

## Key Artifacts to Produce
- `src/submission/__init__.py`: Submission package exports.
- `src/submission/exporter.py`: `SubmissionExporter` compiling self-contained offline inference script and notebook.
- `src/submission/validator.py`: `SubmissionValidator` checking `submission.csv` formatting and profiling execution latency.
- `src/submission/metadata.py`: `generate_kernel_metadata` creating `kernel-metadata.json` for Kaggle API push.
- `tests/test_submission_exporter.py`: Unit tests verifying offline script compilation, importability, and self-containment.
- `tests/test_submission_validator.py`: Unit and integration tests verifying schema validation, NaN detection, probability bounds, and throughput benchmarks.
