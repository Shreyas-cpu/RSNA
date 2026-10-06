---
phase: 5
slug: clinical-radiology-report-weak-supervision-engine
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-10-06
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_report_parser.py tests/test_weak_validator.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_report_parser.py tests/test_weak_validator.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | WEAK-01 | — | Deterministic NegEx parser & entity extractor | unit | `pytest tests/test_report_parser.py` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | WEAK-01 | — | Structured prompt & JSON response parsing | unit | `pytest tests/test_llm_extractor.py` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 2 | WEAK-02 | — | Ground-truth alignment verification & silver dataset | integration | `pytest tests/test_weak_validator.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_report_parser.py` — unit tests for NegEx scoping, certainty levels, and 12-class extraction
- [ ] `tests/test_llm_extractor.py` — unit tests for prompt formatting and JSON extraction parsing
- [ ] `tests/test_weak_validator.py` — integration tests for ground-truth alignment metrics and silver dataset generation

---

## Validation Signoff

- [ ] All automated tests passing
- [ ] WEAK-01 and WEAK-02 verified
- [ ] Ready for Phase 6 multi-plane feature fusion
