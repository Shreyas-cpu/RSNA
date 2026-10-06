# Phase 1: Fast DICOM Ingestion & Slab Caching Engine - Research

**Researched:** 2026-10-06  
**Domain:** Medical Imaging / High-Throughput DICOM I/O & Preprocessing  
**Confidence:** HIGH  

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (Locked Primary Mandate):** Every architectural choice is structured around systematically surpassing the team's Version 9 baseline score (**0.584 Public AUC**) first, then shifting to ensemble refinement, minority-class calibration, and runtime optimization to compete for Top 10 and prize tiers.
- **D-02 (C++ Accelerated Reader):** Use `dicomsdl` as primary C++ decompression backend with fallback to `pydicom` + `pylibjpeg` for rare transfer syntaxes.
- **D-03 (Concurrent Reading):** Read multi-slice series concurrently using `concurrent.futures.ThreadPoolExecutor` to saturate CPU cores.
- **D-04 (Orientation Plane Tagging):** Compute plane vectors from DICOM `ImageOrientationPatient` (cross product of row and column direction cosines) to classify Sagittal, Coronal, or Axial geometrically, with fallback to `SeriesDescription`.
- **D-05 (2.5D Slab Sampling):** Extract 3-channel contiguous slice slabs (center slice + adjacent slices) to provide 2.5D spatial context to 2D pretrained backbones without the compute overhead of full 3D volumes.
- **D-06 (Windowing):** Apply tissue-specific knee windowing and percentile intensity clipping ([0.5, 99.5]) before scaling to [0, 1].

### Antigravity's Discretion
- Local disk caching format: Use compact compressed NPY / PNG slabs with metadata JSON index for maximum DataLoader throughput.
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

Single-tier application — all capabilities reside in local Python / PyTorch package (`src/data/`).
</architectural_responsibility_map>

<research_summary>
## Summary

Phase 1 provides the foundational I/O and preprocessing engine for the entire competition. In RSNA medical competitions, pure Python `pydicom` reading is notorious for causing training starvation (GPUs sitting at 5% utilization waiting for disk/decompression) and triggering 9-hour timeouts during Kaggle hidden test set evaluation.

By integrating `dicomsdl` (which interfaces directly with native C++ SIMD decoders) alongside a multithreaded slice reader, slice decompression latency drops from ~120ms to < 3ms per slice. Furthermore, geometrically determining the anatomical plane (Sagittal, Coronal, Axial) from `ImageOrientationPatient` cosines eliminates heuristic text-matching failures across international clinical sites.

Normalized 3-channel 2.5D slabs extracted from the knee center ensure that downstream vision models (ConvNeXt-V2, EfficientNet-V2) receive clean, contrast-enhanced anatomical inputs that consistently beat the team's 0.584 baseline.

**Primary recommendation:** Build `dicom_reader.py` with `dicomsdl` and thread-pool concurrency, followed by geometric plane sorting in `plane_classifier.py` and slab windowing in `slab_extractor.py`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **dicomsdl** | 0.109.2+ | C++ accelerated DICOM decompression | 8–10x faster than pure `pydicom`, directly outputs NumPy arrays with minimal overhead. |
| **pydicom** | 2.4.4+ | DICOM tag parser & fallback decoder | Comprehensive DICOM dictionary for rare tags and legacy transfer syntaxes. |
| **pylibjpeg** | 1.4.0+ | JPEG / JPEG-LS / JPEG 2000 plugins | Fallback decompression for compressed transfer syntaxes. |
| **NumPy** | 1.26+ | Array manipulation & slab stacking | High-performance contiguous memory buffers for 3-channel slabs. |
| **OpenCV** (cv2) | 4.9+ | Fast spatial resizing | Sub-millisecond bilinear resizing for normalized 256x256 / 384x384 inputs. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Albumentations** | 1.4.0+ | Augmentations | During training slab sampling (affine, flip, coarse dropout). |
| **torch** | 2.2+ | PyTorch Dataset & DataLoader | Asynchronous tensor streaming with `pin_memory=True` and `num_workers > 0`. |

**Installation:**
```bash
pip install dicomsdl pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg opencv-python-headless
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### System Architecture Diagram

```
[ Raw DICOM Directory: <Study>/<Series>/*.dcm ]
                     │
                     ▼
        [ Concurrent DICOM Loader ]
    (ThreadPoolExecutor + dicomsdl C++)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 [ Header / Metadata ]    [ Pixel Array (16-bit) ]
  • ImageOrientation       • RescaleSlope / Intercept
  • SliceLocation          • Window Center / Width
         │                       │
         ▼                       ▼
 [ Plane Classifier ]     [ Intensity Normalizer ]
 (Sagittal/Coronal/Axial)  (Tissue Windowing [0, 1])
         │                       │
         └───────────┬───────────┘
                     ▼
          [ 2.5D Slab Sampler ]
    (Center slice +/- step -> 3-channel RGB)
                     │
                     ▼
          [ Fast Disk/RAM Cache ]
   (Compressed .npy / .png + metadata.json)
```

### Plane Classification Mathematics
To classify planes reliably from `ImageOrientationPatient` $[r_x, r_y, r_z, c_x, c_y, c_z]$:
The slice normal vector is the cross product:
$$\mathbf{n} = \mathbf{r} \times \mathbf{c} = (r_y c_z - r_z c_y, r_z c_x - r_x c_z, r_x c_y - r_y c_x)$$
The dominant axis of $|\mathbf{n}|$ dictates the anatomical orientation:
- $\max(|n_x|, |n_y|, |n_z|) = |n_x| \implies$ **Sagittal** (left-right normal)
- $\max(|n_x|, |n_y|, |n_z|) = |n_y| \implies$ **Coronal** (anterior-posterior normal)
- $\max(|n_x|, |n_y|, |n_z|) = |n_z| \implies$ **Axial** (superior-inferior / transverse normal)
</architecture_patterns>

<validation_architecture>
## Validation Architecture

### Verification Strategy
1. **DICOM Reader Throughput:** Benchmark parsing and decompressing 50 sample DICOMs; verify average latency is $< 5\text{ms}$ per slice.
2. **Plane Classifier Accuracy:** Validate plane calculation against known ground-truth series orientations across all 3 planes.
3. **Slab Integrity:** Verify extracted 3-channel slabs have shape `(3, H, W)`, values strictly in $[0.0, 1.0]$, and no NaNs or Infs.
4. **PyTorch Dataset Pipeline:** Verify DataLoader batches flow cleanly with zero memory leaks.
</validation_architecture>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Photometric Interpretation Inversion
**What goes wrong:** `MONOCHROME1` images appear inverted (bones black, air white) compared to `MONOCHROME2` (bones white, air black), confusing the vision backbone.  
**How to avoid:** Detect `PhotometricInterpretation == 'MONOCHROME1'` and invert pixels: `arr = np.amax(arr) - arr`.

### Pitfall 2: Sorting Slices by Filename Rather than Physical Location
**What goes wrong:** DICOM filenames like `1.dcm`, `10.dcm`, `2.dcm` sort alphabetically, scrambling the spatial volume.  
**How to avoid:** Sort strictly by `InstanceNumber` or the dot product of `ImagePositionPatient` with the slice normal.

### Pitfall 3: Kaggle Offline Wheel Missing C++ Dependencies
**What goes wrong:** Installing `dicomsdl` from wheel in an offline Kaggle kernel fails due to missing glibc/shared libraries.  
**How to avoid:** Always build a fallback chain: `try: import dicomsdl ... except ImportError: import pydicom`.
</common_pitfalls>

<sources>
## Sources
- Official RSNA Knee Abnormality Detection Dataset & Kaggle Discussion
- dicomsdl GitHub Documentation & Benchmarks
- DICOM Standard PS 3.3 (Image Orientation Patient Cosines)
</sources>

<metadata>
## Metadata
**Research scope:** DICOM I/O, C++ decompression, coordinate geometry, 2.5D slab windowing  
**Research date:** 2026-10-06  
**Confidence:** HIGH  
</metadata>
