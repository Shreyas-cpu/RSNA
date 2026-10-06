# Stack Research

**Domain:** Medical Imaging / Deep Learning (Multimodal Knee MRI & Clinical NLP)  
**Researched:** 2026-10-06  
**Confidence:** HIGH  

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

```bash
# Core Machine Learning & Medical Imaging
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scikit-learn pydicom dicomsdl pylibjpeg

# Tabular & NLP
pip install polars pandas transformers accelerate
```

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

**If training locally (RTX 4060 8GB):**
- Use PyTorch AMP (`torch.cuda.amp.autocast(dtype=torch.float16)`).
- Use gradient accumulation (effective batch size 32, micro batch size 8).
- Cache 2.5D 3-channel slabs at 256x256 or 384x384.

**If exporting to Kaggle submission kernel:**
- Bundle all dependency wheels (`dicomsdl`, `timm`) in a private Kaggle dataset.
- Mount model weights as read-only dataset inputs.
- Keep kernel self-contained with zero internet access dependencies.

## Sources

- Official Kaggle RSNA Knee Abnormality Detection Competition Environment
- PyTorch & timm Documentation
- dicomsdl benchmarks on RSNA medical imaging competitions

---
*Stack research for: RSNA Knee Abnormality Detection Pipeline*  
*Researched: 2026-10-06*
