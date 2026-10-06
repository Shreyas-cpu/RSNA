# Phase 7 Research: Kaggle Submission Environment & Efficiency Optimization

## 1. Kaggle Offline Execution & Directory Hierarchy

Kaggle code competitions run in an air-gapped container with network isolation:
```text
/kaggle/
├── input/
│   ├── rsna-knee-abnormality-detection/
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── test_series/
│   │       └── <StudyInstanceUID>/
│   │           └── <SeriesInstanceUID>/
│   │               ├── 1.dcm
│   │               ├── 2.dcm
│   │               └── ...
│   ├── rsna-knee-weights/             <-- Custom dataset with best_model.pt
│   └── rsna-knee-wheels/              <-- Custom dataset with offline pip wheels
└── working/
    └── submission.csv                 <-- Evaluated output artifact
```

### Wheel Management Without Internet
To satisfy `enable_internet: false`:
```bash
pip install --no-index --find-links=/kaggle/input/rsna-knee-wheels timm
```
Alternatively, our pipeline can leverage built-in torchvision ResNet/EfficientNet backbones to run with zero extra pip wheel dependencies, eliminating installation failure points.

---

## 2. $17,000 Model Efficiency Track Requirements

The Model Efficiency Prize is awarded to top-performing submissions that run within strict compute bounds:
1. **Runtime Limit**: Inference over the entire hidden test set must complete in **< 30 minutes** on standard Kaggle GPU runtimes (T4 x 2 or P100).
2. **Key Optimizations**:
   - `torch.inference_mode()`: Bypasses autograd engine overhead.
   - `torch.autocast('cuda', dtype=torch.float16)`: 2x faster tensor core throughput with 50% memory footprint.
   - Dynamic series sampling: Extracting 4 representative slabs per plane rather than processing 60 redundant slices.
   - High-throughput DICOM loader: Direct memory mapping and fast orientation cosine plane classification (< 2.5 ms/slice).

---

## 3. Strict Schema Compliance & Error Guards

A submission error on Kaggle exhausts one of only 5 daily submission attempts. The submission validator enforces:
1. **Column Names**: Exactly matches `sample_submission.csv` (`StudyInstanceUID` + 12 target columns).
2. **Row Count & Ordering**: Exactly matching `test.csv`.
3. **Value Range**: Every prediction strictly bounded in $[0.0, 1.0]$.
4. **NaN / Null Guard**: Automatic fallback imputation with baseline class priors ($p \approx 0.10$) if a corrupted DICOM series fails to load.

---

## 4. Kaggle CLI Integration

`kernel-metadata.json` standard format enables direct deployment via `kaggle kernels push -p submission/`:
```json
{
  "id": "promethious/rsna-knee-submission",
  "title": "RSNA Knee Multi-Plane Attention Submission",
  "code_file": "submission.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "competition_sources": [
    "rsna-knee-abnormality-detection"
  ]
}
```
