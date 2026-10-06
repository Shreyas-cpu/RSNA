# Architecture Research

**Domain:** Medical Imaging / Deep Learning (RSNA Knee Abnormality Detection)  
**Researched:** 2026-10-06  
**Confidence:** HIGH  

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION & PREPROCESSING                      │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌─────────────────────┐    ┌────────────────┐  │
│  │ Fast DICOM Reader│───▶│ Plane Classifier    │───▶│ 2.5D Slab Gen  │  │
│  │ (dicomsdl / C++) │    │ (Sag/Cor/Axial)     │    │ (Norm/Window)  │  │
│  └──────────────────┘    └─────────────────────┘    └────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                     WEAK SUPERVISION SUBSYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ LLM Report Parser: Unstructured Text ──▶ 12-Class Pseudo-Labels  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│                     DEEP LEARNING MODEL PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌─────────────────────┐    ┌────────────────┐  │
│  │ 2D Slice Encoder │───▶│ MIL Attention Pool  │───▶│ 12-Target Head │  │
│  │ (ConvNeXt/EffNet)│    │ (Gated Attention)   │    │ (Sigmoid Prob) │  │
│  └──────────────────┘    └─────────────────────┘    └────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                     EVALUATION & KERNEL EXPORT                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────┐  ┌─────────────────────────────┐  │
│  │ Stratified Group 5-Fold Evaluator│  │ Kaggle Offline Kernel Pack  │  │
│  │ (Macro AUC ROC Metric Tracker)   │  │ (sub-15 min, submission.csv)│  │
│  └──────────────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Data Loader & Preprocessor** | Decompresses DICOM slices, parses plane metadata, handles windowing, extracts central slices/slabs. | `dicomsdl` + Python multi-threading, custom PyTorch Dataset with Albumentations. |
| **Weak-Label Generator** | Ingests unannotated clinical radiology impressions, queries LLM with structured prompt, outputs 12 continuous probabilities. | Python script with Gemini / local LLM, writing to `silver_train.csv`. |
| **Vision Backbone** | Extracts dense spatial feature vectors from individual slices (2.5D: 3 consecutive slices as RGB channels). | `timm.create_model('convnext_small.fb_in22k_ft_in1k_384', pretrained=True)`. |
| **Sequence Aggregator** | Aggregates $N$ slice vectors into a single study-level embedding using attention weights. | Gated Attention / Multi-Head Self-Attention Pooling. |
| **Multi-Label Classification Head** | Maps the aggregated study embedding to 12 abnormality logit predictions. | Linear Layer (dim $\rightarrow$ 12) + Sigmoid. |
| **Inference Exporter** | Packages model weights, wheel dependencies, and inference kernel into a single self-contained Kaggle submission notebook. | Python export utility validating execution on sample test set. |

## Recommended Project Structure

```
RSNA/
├── .planning/                       # GSD planning state & roadmaps
├── about_RSNA.md                    # Comprehensive competition guide
├── configs/                         # Modular YAML experiment configs
│   ├── baseline_convnext.yaml       # Fast 2.5D ConvNeXt configuration
│   └── efficientnet_v2.yaml         # Lightweight architecture configuration
├── data/                            # Raw data & extracted cache directories
│   ├── raw/                         # Raw DICOMs (train_series, test_series)
│   ├── processed/                   # Preprocessed slabs / HDF5 cache
│   └── silver_labels/               # LLM-extracted pseudo labels
├── src/                             # Core Python package
│   ├── data/
│   │   ├── dicom_reader.py          # High-speed dicomsdl & pydicom wrapper
│   │   ├── dataset.py               # PyTorch Dataset for 2.5D slab loading
│   │   └── transforms.py            # Augmentation pipelines
│   ├── models/
│   │   ├── backbone.py              # timm vision feature extractor
│   │   ├── attention_pool.py        # Multiple Instance Learning pooling
│   │   └── classifier.py            # End-to-end multi-plane study model
│   ├── nlp/
│   │   └── report_extractor.py      # LLM prompt & structured weak-label parser
│   ├── training/
│   │   ├── losses.py                # Class-balanced BCE / Focal loss
│   │   ├── trainer.py               # AMP training loop with checkpointing
│   │   └── metrics.py               # Macro-averaged ROC AUC calculation
│   └── submission/
│       └── kernel_builder.py        # Kaggle offline submission generator
├── notebooks/                       # EDA and exploratory notebooks
└── tests/                           # Unit tests for preprocessing & metric calculation
```

## Data Flow & Build Order

1. **Step 1: Ingestion & Fast DICOM Reader**: Build `dicom_reader.py` first to benchmark slice decompression speed.
2. **Step 2: Dataset & Slab Caching**: Implement 2.5D slab extraction and plane categorization.
3. **Step 3: Vision Backbone & Pooling Head**: Implement `backbone.py` and `attention_pool.py` and run dummy forward passes to verify shapes.
4. **Step 4: Training & Local Validation**: Set up Stratified Group K-Fold cross-validation and verify local Macro AUC metric computation.
5. **Step 5: NLP Weak Label Generator**: Generate silver labels to expand training data volume.
6. **Step 6: Submission Kernel Export**: Automatically generate and verify the offline Kaggle submission notebook to guarantee valid submissions.

---
*Architecture research for: RSNA Knee Abnormality Detection Pipeline*  
*Researched: 2026-10-06*
