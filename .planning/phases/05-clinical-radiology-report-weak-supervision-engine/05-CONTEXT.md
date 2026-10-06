# Phase 5 Context: Clinical Radiology Report Weak Supervision Engine

## Overview
Phase 5 develops the clinical radiology NLP extraction engine to harvest high-quality, structured 12-class pseudo-probabilities $[0.0, 1.0]$ from narrative radiology impression texts in `train.csv`. In the RSNA 2026 challenge, only a fraction of MRI studies have verified ground-truth binary labels; the remainder include unstructured diagnostic reports. This phase provides the weak supervision engine that unlocks 3–5x more training cases, directly boosting vision model representations and driving Macro AUC past the 0.584 baseline.

## Decisions

- **D-17:** Dual-Engine Clinical Report Extraction (WEAK-01) — Build `ClinicalReportParser` combining an optimized clinical regex / NegEx negation-scoping rule parser for instant offline evaluation, alongside an extensible `LLMReportExtractor` structured prompt interface. For each of the 12 target pathologies, the parser outputs continuous calibrated probabilities in $[0.0, 1.0]$ reflecting diagnostic certainty (e.g., negative/intact $\to 0.05$, suspected/equivocal $\to 0.50$, definitive tear/abnormality $\to 0.95$).
- **D-18:** Ground-Truth Alignment Harness & Silver Dataset Generation (WEAK-02) — Implement `WeakLabelValidator` to evaluate pseudo-label quality against the subset of studies with expert human ground-truth annotations. Verify $>90\%$ alignment / AUC against ground truth, calibrate confidence thresholds, and generate `silver_train.csv` annotated with sample weights for downstream vision fine-tuning.

## Key Artifacts to Produce
- `src/nlp/__init__.py`: NLP package exports.
- `src/nlp/report_parser.py`: `ClinicalReportParser` with clinical entity vocabularies, NegEx negation scoping, and certainty scoring for all 12 pathologies.
- `src/nlp/llm_extractor.py`: Structured prompt template and JSON extraction engine for zero-shot / few-shot LLM inference.
- `src/nlp/validator.py`: `WeakLabelValidator` benchmarking pseudo-labels against ground truth (ROC AUC, correlation, F1) and generating `silver_train.csv`.
- `tests/test_report_parser.py`: Unit tests for negation detection, synonym mapping, certainty scoring, and edge cases.
- `tests/test_llm_extractor.py`: Unit tests for structured JSON prompt rendering and output parsing.
- `tests/test_weak_validator.py`: Unit and integration tests for validation alignment metrics and silver dataset generation.
