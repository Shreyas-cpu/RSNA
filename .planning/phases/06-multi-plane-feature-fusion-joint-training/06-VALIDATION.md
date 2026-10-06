---
phase: 6
slug: multi-plane-feature-fusion-joint-training
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-06
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_multi_plane.py tests/test_joint_trainer.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~16 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_multi_plane.py tests/test_joint_trainer.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | MODEL-03 | — | Multi-plane attention fusion & missing plane handling | unit | `pytest tests/test_multi_plane.py` | ✅ | ✅ green |
| 06-02-01 | 02 | 2 | TRAIN-03 | — | Sample-weighted BCE loss & joint gold/silver curriculum | unit/int | `pytest tests/test_joint_trainer.py` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_multi_plane.py` — unit tests for multi-plane forward pass, plane masking, and gradient propagation
- [x] `tests/test_joint_trainer.py` — unit tests for sample-weighted multi-label loss and joint curriculum training

---

## Validation Signoff

- [x] All automated tests passing (70/70 green)
- [x] MODEL-03 and TRAIN-03 verified
- [x] Ready for Phase 7 offline Kaggle kernel exporter
