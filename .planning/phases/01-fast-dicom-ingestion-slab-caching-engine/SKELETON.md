# Walking Skeleton — RSNA Knee Abnormality Detection Pipeline

**Phase:** 1  
**Generated:** 2026-10-06  

## Capability Proven End-to-End

A knee MRI study folder containing multi-slice DICOM series is ingested by the C++ accelerated reader, classified into Sagittal/Coronal/Axial anatomical planes, windowed for knee joint tissue contrast, and converted into a normalized 3-channel 2.5D slab tensor ready for PyTorch vision backbones in < 5ms per slice.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **DICOM Engine** | `dicomsdl` (C++) + `pydicom` | 8-10x throughput speedup over pure Python; prevents GPU starvation and guarantees sub-15 minute test inference for the Efficiency Track. |
| **Orientation Geometry** | Normal vector from direction cosines | Cross product of row and column cosines ($\mathbf{r} \times \mathbf{c}$) determines anatomical plane without fragile metadata string heuristics. |
| **Volumetric Dimension** | 2.5D contiguous 3-slice slab | Delivers inter-slice context to pretrained 2D CNN/Transformer backbones without the compute and VRAM explosion of full 3D CNNs. |
| **Tissue Normalization** | Knee windowing + percentile clipping | Standardizes high-dynamic-range 16-bit MR signal intensity into $[0.0, 1.0]$ across multi-vendor scanners. |
| **Cache Format** | Compressed NPY/PNG + metadata JSON | Zero-overhead streaming for PyTorch DataLoader, keeping GPU compute saturated at 100%. |

## Stack Touched in Phase 1

- [ ] Project scaffold (`src/`, `tests/`, `pyproject.toml`)
- [ ] DICOM Reader (`dicomsdl` with fallback to `pydicom` + `pylibjpeg`)
- [ ] Plane Classifier (`ImageOrientationPatient` cross product normal vector)
- [ ] 2.5D Slab Extractor (windowing, percentile clipping, 3-slice stack)
- [ ] Dataset & Cache Streamer (PyTorch `Dataset` streaming cached slabs)

## Out of Scope (Deferred to Later Slices)

- Multi-slice sequence attention pooling head (Phase 3)
- Multi-plane concatenation fusion (Phase 6)
- Radiology report NLP extraction (Phase 5)
- Kaggle kernel export (Phase 7)

## Subsequent Slice Plan

- **Phase 2:** Stratified Group 5-Fold validation harness and exact Macro AUC metric evaluator.
- **Phase 3:** 2.5D Multi-Slice feature extractor (ConvNeXt-V2 / EfficientNet-V2) with MIL Attention Pooling.
- **Phase 4:** Mixed-precision training loop beating the 0.584 baseline score locally.
- **Phase 5:** LLM weak-label extractor parsing narrative radiology reports into silver training data.
- **Phase 6:** Multi-plane fusion model trained jointly on gold ground truth and silver labels.
- **Phase 7:** Offline Kaggle submission kernel and Efficiency Track runtime optimization.
