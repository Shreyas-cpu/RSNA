# RSNA Knee Abnormality Detection (2026) — Comprehensive Competition Guide

> **Competition Link:** [kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Host:** Radiological Society of North America (RSNA) in partnership with Kaggle  
> **Total Prize Pool:** $77,000 USD (Accuracy Track: $60,000 | Model Efficiency Track: $17,000)  
> **Evaluation Metric:** Macro-averaged Area Under the ROC Curve (Macro AUC ROC)  
> **Current Team Baseline:** `rsna-knee-submission-baseline` (Version 9) — **Public Score: 0.584**  

---

## 1. Overview Tab

### Clinical Motivation & Challenge Goal
Knee pain and joint degeneration are among the most common musculoskeletal complaints worldwide. While Magnetic Resonance Imaging (MRI) is the gold standard diagnostic tool, interpreting complex knee exams is time-intensive, subject to inter-observer variability, and hindered by global shortages of specialized musculoskeletal (MSK) radiologists.

The **RSNA Knee Abnormality Detection** challenge tasks participants with developing machine learning models to detect **12 clinically important knee abnormalities** across multiple joint structures at the study level.

### The Paradigm Shift: Multimodal Learning with Real-World Clinical Reports
This competition represents a breakthrough in medical AI challenges:
- **Prior RSNA Competitions** relied strictly on curated bounding boxes, slice annotations, or isolated volumetric scans.
- **2026 Challenge Innovation:** Pairs multi-series MRI studies directly with **original, unstructured, multilingual clinical radiology reports**.
- Only a subset of training exams feature verified ground-truth binary labels; the remainder include diagnostic narratives. Top solutions must bridge the gap between computer vision and clinical NLP (e.g., using LLM report extraction and weak supervision).

---

## 2. Data Tab

### Dataset Scale & Heterogeneity
- **Cohort:** >5,000 knee MRI examinations sourced from **19 international clinical sites** across diverse patient demographics, scanner manufacturers (GE, Siemens, Philips), field strengths (1.5T, 3.0T), and imaging protocols.
- **Volumetric Modality:** Native DICOM format organized hierarchically:
  ```
  train_series/
    ├── <StudyInstanceUID>/
    │     ├── <SeriesInstanceUID>/
    │     │     ├── <SOPInstanceUID_1>.dcm
    │     │     ├── <SOPInstanceUID_2>.dcm
    │     │     └── ...
  test_series/
    ├── <StudyInstanceUID>/
    │     ├── <SeriesInstanceUID>/
    │     │     └── ...
  ```

### Key Files & Schemas

#### 1. `train.csv`
Contains study-level metadata, radiology reports, and ground truth annotations:
- `StudyInstanceUID`: Unique identifier for each MRI examination.
- `Report`: Unstructured free-text radiology impression and findings (multilingual / multi-institution).
- **12 Target Binary Labels (0 or 1)**:
  1. `ACL` — Anterior Cruciate Ligament tear / rupture
  2. `MCL` — Medial Collateral Ligament tear / sprain
  3. `Medial Meniscus` — Medial meniscal tear
  4. `Lateral Meniscus` — Lateral meniscal tear
  5. `Medial OA` — Medial compartment osteoarthritis / cartilage loss
  6. `Lateral OA` — Lateral compartment osteoarthritis / cartilage loss
  7. `PF OA` — Patellofemoral compartment osteoarthritis
  8. `Effusion` — Joint effusion / fluid accumulation
  9. `Synovitis` — Synovial thickening / inflammatory pannus
  10. `Baker's` — Baker's (popliteal) cyst
  11. `Contusion` — Trabecular microfracture / bone marrow lesion / bone bruise
  12. `Fracture` — Cortical or subchondral bone fracture

*Note: For exams where only the `Report` is provided without manual ground truth, target columns are null/unlabeled, providing a large-scale weak supervision / semi-supervised learning opportunity.*

#### 2. `test.csv`
Contains the test set study metadata (`StudyInstanceUID`) to be evaluated on Kaggle's hidden test set.

#### 3. `sample_submission.csv`
Defines the required output format:
```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
test_study_001,0.05,0.02,0.85,0.12,0.40,0.08,0.22,0.91,0.65,0.10,0.04,0.01
...
```

#### 4. Series Metadata & Imaging Characteristics
Each study includes multiple pulse sequences and orientations:
- **Anatomical Planes:** Sagittal (crucial for ACL, PCL, menisci), Coronal (collateral ligaments, meniscal bodies, tibial plateaus), and Axial (patellofemoral joint, retinacula).
- **Contrast / Weighting:**
  - `Fluid_Sensitive`: T2-weighted / Proton Density (PD) fat-suppressed sequences (highlights fluid, edema, tears, effusion).
  - Non-fluid-sensitive / T1-weighted: Delineates anatomy, subchondral sclerosis, and bone boundaries.

---

## 3. Evaluation & Leaderboard Tab

### Evaluation Metric: Macro-Averaged AUC ROC
The official evaluation metric is the unweighted mean of the Area Under the Receiver Operating Characteristic Curve (ROC AUC) computed across all 12 individual targets:

$$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{AUC}_i$$

- **Probabilistic Predictions:** All target values in `submission.csv` must be continuous probabilities in $[0.0, 1.0]$.
- **Macro Averaging Impact:** Rare conditions (e.g., `Fracture`, `Contusion`) carry equal weighting to common conditions (`Effusion`, `Medial Meniscus`). Calibration and high sensitivity/specificity across minority classes are vital.

### Leaderboard Mechanics
- **Public Leaderboard:** Calculated on a subset (~25–30%) of the hidden test data to provide real-time feedback during the competition.
- **Private Leaderboard:** Evaluated on the remaining hidden test data (~70–75%) to determine final official winners.
- **Submission Limits:** Maximum **5 submissions per day** per team. Up to **2 final submissions** can be selected for final scoring.

---

## 4. Current Team Baseline & Progression

From our baseline development notebook (`rsna-knee-submission-baseline`), we have tracked the iterative submission performance on the public leaderboard:

| Version | Status | Public AUC Score | Notes & Milestones |
| :--- | :--- | :--- | :--- |
| **Version 3** | Succeeded | **0.497** | Initial submission pipeline & dummy baseline verification |
| **Version 5** | Succeeded | **0.496** | Initial DICOM loading verification |
| **Version 7** | Succeeded | **0.533** | First active vision feature extraction |
| **Version 9** | Succeeded | **0.584** (Current Best) | **Champion Baseline**: Working multi-slice extraction producing 0.584 AUC |
| **Version 11** | Succeeded | **0.567** | Experimental variation (slight drop; indicates sensitivity to pooling/aggregation) |

**Target Milestone:** Beat **0.584** $\rightarrow$ push towards **0.65+** through systematic 2.5D multi-plane attention, weak supervision from reports, and ensemble calibration.

---

## 5. Kaggle Notebook Lifecycle Structure (Notebook, Input, Output, Logs, Comments)

When engineering the pipeline to reproduce and surpass Version 9, every notebook artifact is aligned with Kaggle's platform lifecycle:

1. **Notebook Tab (`Notebook`)**:
   - Contains self-contained, clean inference code:
     - Dependency imports (wheels from input datasets).
     - Fast DICOM reader (`dicomsdl` / `libjpeg-turbo`).
     - Image preprocessor: dynamic windowing, plane grouping (Sagittal, Coronal, Axial), slab sampling.
     - Model forward pass with PyTorch AMP (fp16) inference.
     - Target calibration and `submission.csv` generation.

2. **Input Tab (`Input`)**:
   - Contains mounted read-only Kaggle datasets:
     - `/kaggle/input/rsna-knee-abnormality-detection/` (competition data: test series, sample submission).
     - Trained model checkpoints dataset (e.g. `rsna-knee-models-v9`, `rsna-knee-convnext-checkpoints`).
     - Pre-installed wheels dataset (e.g. offline `dicomsdl`, `timm`, `albumentations` packages).

3. **Output Tab (`Output`)**:
   - Must output `/kaggle/working/submission.csv` strictly formatted with `StudyInstanceUID` and the 12 predicted probability columns.
   - Verification assert check: ensure no NaNs, correct row count matching `test.csv`, and probabilities bounded in $[0.0, 1.0]$.

4. **Logs Tab (`Logs`)**:
   - Diagnostic stdout/stderr:
     - Timing per study / batch (verifying throughput against the 9-hour limit and qualifying for the **Efficiency Track**).
     - GPU memory allocation peaks (avoiding CUDA OOM).

5. **Comments Tab (`Comments`)**:
   - Used for documentation of experiments, architecture diffs between versions, and hyperparameter logs.

---

## 6. Prizes & Tracks

### Total Prize Pool: $77,000 USD

#### Main Accuracy Leaderboard ($60,000 Total)
- **1st Place:** $9,000
- **2nd Place:** $8,000
- **3rd Place:** $7,000
- **4th Place:** $6,500
- **5th Place:** $6,000
- **6th Place:** $5,500
- **7th Place:** $5,000
- **8th Place:** $5,000
- **9th Place:** $4,000
- **10th Place:** $4,000

#### Model Efficiency Track ($17,000 Total)
- **1st Place:** $7,000
- **2nd Place:** $6,000
- **3rd Place:** $5,000
- Evaluated on total notebook runtime (`RuntimeSeconds`) during test inference (including DICOM loading, preprocessing, and model forward passes) while achieving a competitive baseline AUC threshold.

---

## 7. Rules & Code Submission Constraints

### Code Competition Requirements
- **Submission Mode:** Submissions must be made via **Kaggle Notebooks**. Direct CSV uploads are disabled.
- **Runtime Limits:**
  - GPU Inference: Notebooks must complete execution within **9 hours** on Kaggle's standard GPU hardware (NVIDIA T4 / P100).
  - Internet Access: **Disabled** during notebook evaluation against the private test set.
  - Pretrained Weights / Packages: Must be bundled in Kaggle Datasets and mounted locally in the notebook environment.

### External Data & Pre-trained Models
- External pre-trained models and public datasets are permitted **only if they are publicly announced and accessible to all participants on the official Discussion forum** before the deadline.
- Proprietary or restricted medical datasets requiring institutional data-use agreements that exclude other competitors are prohibited.

### Open-Source Licensing
- Winning teams are required to release their code and model weights under an OSI-approved open-source license (MIT, Apache 2.0, etc.) to claim prize awards and present at the RSNA Annual Meeting.

---

## 8. Strategy Roadmap: Beating the 0.584 Baseline

```
[ rsna-knee-submission-baseline (V9: 0.584) ]
                     │
                     ▼
  Phase 1: High-Speed I/O & Preprocessing
     • Integrate dicomsdl / libjpeg-turbo
     • Plane separation (Sagittal / Coronal / Axial)
     • Uniform slab extraction & windowing
                     │
                     ▼
  Phase 2: LLM Weak Label Extraction
     • Extract structured 12-class pseudo-labels from free-text reports
     • Multi-fold pseudo-label training to scale dataset
                     │
                     ▼
  Phase 3: Multi-Plane 2.5D Vision Architecture
     • Backbone: ConvNeXt-Small / EfficientNet-V2 / DINOv2
     • Slices -> MIL Attention Pooling -> 12-class multi-label head
     • Class-imbalance weighted BCEWithLogitsLoss
                     │
                     ▼
  Phase 4: Validation Rigor & Metric Tracking
     • Stratified Group 5-Fold matching Kaggle Macro AUC
     • Target: Local CV > 0.62+ predicting LB > 0.60+
                     │
                     ▼
  Phase 5: Kaggle Kernel Exporter & Ensembling
     • Self-contained offline submission script
     • Multi-fold checkpoint blend + TTA
     • Optimized for the $17,000 Efficiency Prize
```
