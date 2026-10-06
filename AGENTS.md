<!-- GSD:project-start source:PROJECT.md -->

## Project

**RSNA Knee Abnormality Detection Pipeline**

An end-to-end, competitive machine learning pipeline for the Kaggle RSNA Knee Abnormality Detection challenge. It ingests multimodal knee MRI exams (DICOM series across Sagittal, Coronal, and Axial planes) paired with clinical radiology reports, detects 12 target joint abnormalities, and produces high-speed, calibrated predictions that surpass the team's current Version 9 baseline (0.584 Public AUC) while optimizing for the Model Efficiency track.

**Core Value:** Beat the 0.584 Public AUC baseline with a robust, reproducible 2.5D multi-plane attention architecture and generate submission-ready Kaggle kernels under strict runtime limits.

### Constraints

- **Submission Format**: Submissions must run strictly as offline Kaggle Notebooks outputting `submission.csv` with 12 continuous probabilities.
- **Efficiency Constraint**: Total inference runtime on Kaggle hidden test set must be sub-hour (ideally < 15 minutes) to compete for the $17,000 Efficiency Prize.
- **GPU Memory**: Local training and Kaggle inference must fit comfortably within standard VRAM limits (8GB - 16GB).
- **Evaluation Metric**: Unweighted macro average across 12 targets penalizes poor minority-class calibration (`Fracture`, `Contusion`).

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.10+ | Core runtime | Standard runtime for Kaggle GPU environments and PyTorch ecosystem. |
| **PyTorch** | 2.2+ | Deep learning framework | Native mixed precision (AMP fp16/bf16), robust CUDA memory caching, torch.compile support. |
| **timm** | 0.9.16+ | Pretrained vision backbones | State-of-the-art vision models (ConvNeXt-V2, EfficientNet-V2, EVA-02, DINOv2) with exact feature extractors. |
| **dicomsdl** / **pylibjpeg** | 0.109+ | High-speed DICOM parsing | C++ accelerated DICOM decompression; up to 8–10x faster than pure `pydicom`, vital for sub-15 minute test set evaluation. |
| **Albumentations** | 1.4+ | Volumetric & 2D augmentation | GPU/CPU fast spatial & intensity transforms (ShiftScaleRotate, RandomBrightnessContrast, CoarseDropout). |
| **scikit-learn** | 1.4+ | Validation & metric evaluation | `MultilabelStratifiedKFold` / `StratifiedGroupKFold` and precise `roc_auc_score(average='macro')`. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pydicom** | 2.4+ | Fallback DICOM metadata extraction | Extracting study/series tags (`SeriesDescription`, `ImageOrientationPatient`, `SliceLocation`). |
| **transformers** / **vLLM** | 4.38+ | LLM report feature extraction | Extracting 12-class pseudo-labels from narrative radiology impressions. |
| **polars** / **pandas** | 2.0+ | Fast tabular data handling | Indexing `train.csv` metadata and fast aggregation of multi-series mappings. |
| **h5py** / **zarr** | 3.10+ | Local preprocessed slab cache | Ultra-fast caching of normalized 2.5D slabs on local SSD/NVMe for zero-bottleneck training epochs. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **Kaggle CLI** | Automated dataset & kernel sync | `kaggle kernels push` for unattended testing of submission kernels. |
| **Rich / TQDM** | CLI progress & throughput logging | Tracks slices/second and memory footprint across training runs. |

## Installation

# Core Machine Learning & Medical Imaging

# Tabular & NLP

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **2.5D Multi-Slice + MIL Attention** | 3D Dense ResNet/UNet (MONAI) | Only if voxel spacing is strictly uniform and infinite compute is available. Full 3D is too slow and VRAM-heavy for fast iterations. |
| **dicomsdl** | Standard pydicom | `pydicom` is great for debugging tags, but `dicomsdl` is mandatory for competitive inference runtime. |
| **ConvNeXt-V2 / EfficientNet-V2** | Vision Transformer (ViT-H) | ViT-H requires prohibitive VRAM; ConvNeXt-V2 and EfficientNet-V2 deliver superior AUC-per-watt and inference latency. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Uncached on-the-fly full-resolution DICOM reading in PyTorch DataLoader** | CPU/Disk I/O becomes the severe bottleneck, starving GPU and extending epoch times 10x. | Pre-extract and cache normalized multi-slice slabs (HDF5 or PNG/NPY). |
| **Random train/test split** | Causes severe data leakage between series of the same study/patient, giving false CV scores. | `StratifiedGroupKFold` grouped strictly by `StudyInstanceUID`. |
| **Thresholding probabilities before submission** | Competition metric is continuous ROC AUC. Hard binary rounding (0 or 1) destroys AUC ranking. | Continuous predicted probabilities $[0.0, 1.0]$. |

## Stack Patterns by Variant

- Use PyTorch AMP (`torch.cuda.amp.autocast(dtype=torch.float16)`).
- Use gradient accumulation (effective batch size 32, micro batch size 8).
- Cache 2.5D 3-channel slabs at 256x256 or 384x384.
- Bundle all dependency wheels (`dicomsdl`, `timm`) in a private Kaggle dataset.
- Mount model weights as read-only dataset inputs.
- Keep kernel self-contained with zero internet access dependencies.

## Sources

- Official Kaggle RSNA Knee Abnormality Detection Competition Environment
- PyTorch & timm Documentation
- dicomsdl benchmarks on RSNA medical imaging competitions

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.agents/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
