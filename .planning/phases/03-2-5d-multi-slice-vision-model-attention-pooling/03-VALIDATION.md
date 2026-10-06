---
phase: 3
slug: 2-5d-multi-slice-vision-model-attention-pooling
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-06
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_backbone.py tests/test_attention.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_backbone.py tests/test_attention.py tests/test_model.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | MODEL-01 | — | Clean feature extraction without top head | unit | `pytest tests/test_backbone.py` | ✅ Yes | ✅ green |
| 03-02-01 | 02 | 2 | MODEL-02 | — | Attention weights sum to 1.0, shape reduction | unit | `pytest tests/test_attention.py` | ✅ Yes | ✅ green |
| 03-02-02 | 02 | 2 | MODEL-04 | — | 12 continuous probabilities, backward & AMP | integration | `pytest tests/test_model.py` | ✅ Yes | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_backbone.py` — unit tests for `SliceBackbone` across `timm` models
- [x] `tests/test_attention.py` — unit tests for `GatedAttentionPool` weights and dimensionality reduction
- [x] `tests/test_model.py` — integration tests for `RSNAKneeModel` forward pass, gradient flow, and AMP compatibility

---

## Validation Signoff

- [x] All automated tests passing (38/38 green)
- [x] MODEL-01, MODEL-02, and MODEL-04 verified
- [x] Ready for Phase 4 class-balanced training

