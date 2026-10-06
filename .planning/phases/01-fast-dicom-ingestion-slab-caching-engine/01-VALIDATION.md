---
phase: 1
slug: fast-dicom-ingestion-slab-caching-engine
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-06
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_dicom_reader.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_dicom_reader.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | INGEST-01 | — | Safe fallback on corrupt DICOMs | unit | `pytest tests/test_dicom_reader.py -k test_dicom_decompression` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | INGEST-02 | — | Correct plane vector computation | unit | `pytest tests/test_plane_classifier.py` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 2 | INGEST-03 | — | Pixel intensity bounded in [0, 1] | unit | `pytest tests/test_slab_extractor.py -k test_windowing` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 2 | INGEST-04 | — | 3-channel slab shape & cache integrity | integration | `pytest tests/test_dataset_cache.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_dicom_reader.py` — unit tests for `dicomsdl` & `pydicom` reading
- [ ] `tests/test_plane_classifier.py` — unit tests for orientation cosine math
- [ ] `tests/test_slab_extractor.py` — tests for 2.5D slab extraction and windowing
- [ ] `tests/test_dataset_cache.py` — tests for disk caching & DataLoader streaming

---

## Manual-Only Verifications

| Behavior | Why Manual | Verification Procedure |
|----------|------------|------------------------|
| DICOM decompression benchmark | Hardware dependent | Run `python -m src.data.benchmark` and verify average slice read time < 5ms |

---

## Validation Signoff

- [ ] All automated tests passing
- [ ] INGEST-01 through INGEST-04 verified
- [ ] Ready for Phase 2 validation harness
