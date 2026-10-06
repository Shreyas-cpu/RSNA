---
phase: 4
slug: class-balanced-training-baseline-v9-reproduction
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-06
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_loss.py tests/test_trainer.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~14 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_loss.py tests/test_trainer.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | TRAIN-01 | — | Balanced gradient scaling, positive weighting | unit | `pytest tests/test_loss.py` | ✅ | ✅ green |
| 04-01-02 | 01 | 1 | TRAIN-02 | — | Mixed precision AMP, gradient accumulation | unit | `pytest tests/test_trainer.py` | ✅ | ✅ green |
| 04-02-01 | 02 | 2 | EVAL-03 | — | 5-fold CV convergence, beats 0.584 baseline | integration | `pytest tests/test_cv_experiment.py` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_loss.py` — unit tests for `compute_pos_weights` and positive-weighted BCE loss
- [x] `tests/test_trainer.py` — unit tests for `Trainer` training/validation steps, accumulation, and checkpoint saving
- [x] `tests/test_cv_experiment.py` — integration test verifying multi-fold execution and Macro AUC convergence

---

## Validation Signoff

- [x] All automated tests passing (44/44 green)
- [x] TRAIN-01, TRAIN-02, and EVAL-03 verified
- [x] Ready for Phase 5 clinical report weak supervision
