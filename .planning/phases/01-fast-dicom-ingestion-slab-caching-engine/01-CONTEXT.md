# Phase 1: Fast DICOM Ingestion & Slab Caching Engine - Context

**Gathered:** 2026-10-06  
**Status:** Ready for planning  

<domain>
## Phase Boundary

Delivers a high-throughput C++ accelerated DICOM reader (`dicomsdl` / `libjpeg-turbo` with `pydicom` fallback), orientation/plane categorization (Sagittal, Coronal, Axial), tissue windowing, and 2.5D slab extraction and caching engine. Covers requirements INGEST-01, INGEST-02, INGEST-03, INGEST-04.
</domain>

<decisions>
## Implementation Decisions

### Strategic Objective Alignment
- **D-01 (Locked Primary Mandate):** The entire pipeline is engineered around systematically surpassing the team's Version 9 baseline score (**0.584 Public AUC**). High-speed DICOM reading and standardized slab caching are the critical bedrock: without zero-bottleneck I/O, rapid training iterations and submission runtime limits (< 15-30 min for the $17,000 Efficiency Track) cannot be met.

### High-Throughput DICOM Reader (`dicomsdl` vs `pydicom`)
- **D-02:** Use `dicomsdl` as the primary C++ decompression backend with fallback to `pydicom` + `pylibjpeg` for rare transfer syntaxes.
- **D-03:** Read multi-slice series concurrently using `concurrent.futures.ThreadPoolExecutor` to saturate CPU cores during preprocessing and inference.

### Plane Identification & Orientation
- **D-04:** Compute plane vectors from DICOM `ImageOrientationPatient` (cross product of row and column direction cosines) to classify Sagittal, Coronal, or Axial with 100% geometric certainty, falling back to `SeriesDescription` string matching.

### 2.5D Slab Sampling & Windowing
- **D-05:** Extract 3-channel contiguous slice slabs (center slice + adjacent slices) to provide 2.5D spatial context to 2D pretrained backbones without the compute overhead of full 3D volumes.
- **D-06:** Apply tissue-specific knee windowing (window width / window level) and robust percentile intensity clipping ([0.5, 99.5] percentile) before scaling to [0, 1].

### Antigravity's Discretion
- Local disk caching format: Use compact compressed NPY / PNG slabs with metadata JSON index for maximum DataLoader throughput.
</decisions>

<specifics>
## Specific Ideas

- The team's current Version 9 notebook (`rsna-knee-submission-baseline`) proved that basic multi-slice feature extraction achieves 0.584 Public AUC.
- Phase 1 must establish clean, reproducible modular primitives in `src/data/` so that subsequent phases (Validation, 2.5D Attention Model, Training, and Kaggle Exporter) build upon validated, high-speed components.
</specifics>

<canonical_refs>
## Canonical References

- `.planning/PROJECT.md` — Strategic mandate, baseline targets, core constraints
- `about_RSNA.md` — Competition data schemas, planes, and efficiency criteria
- `.planning/research/STACK.md` — `dicomsdl`, `pylibjpeg`, `timm`, `albumentations` versions
- `.planning/research/PITFALLS.md` — Pitfalls 1 & 2 on slow DICOM I/O and Kaggle timeouts
</canonical_refs>
