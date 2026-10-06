---
phase: 2
slug: leakage-free-stratified-group-validation-harness
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-06
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_metrics.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~3 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_metrics.py tests/test_split.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | EVAL-01 | — | Zero group leakage across folds | unit | `pytest tests/test_split.py` | ✅ Yes | ✅ green |
| 02-01-02 | 01 | 1 | EVAL-02 | — | Exact Macro ROC AUC & NaN robustness | unit | `pytest tests/test_metrics.py` | ✅ Yes | ✅ green |
| 02-01-03 | 01 | 1 | EVAL-01, EVAL-02 | — | Full evaluation harness & diagnostics | integration | `pytest tests/test_split.py tests/test_metrics.py -v` | ✅ Yes | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_split.py` — unit tests verifying zero cross-fold group overlap and balanced label ratios
- [x] `tests/test_metrics.py` — unit tests for exact Macro AUC, per-class breakdown, perfect/random scores, and single-class edge cases

---

## Validation Signoff

- [x] All automated tests passing (27/27 green)
- [x] EVAL-01 and EVAL-02 verified
- [x] Ready for Phase 3 vision model & attention pooling

