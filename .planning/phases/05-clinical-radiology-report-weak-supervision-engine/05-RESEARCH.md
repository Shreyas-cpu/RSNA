# Phase 5 Research: Clinical Radiology NLP & Weak Supervision

## 1. Domain Overview & Taxonomy of the 12 Knee Pathologies

Radiology reports for knee MRI typically follow a structured or semi-structured impression format:
```text
FINDINGS:
1. Cruciate Ligaments: Anterior cruciate ligament is completely torn with distal fiber disruption. Posterior cruciate ligament is intact.
2. Collateral Ligaments: Medial collateral ligament demonstrates grade 1 sprain with periligamentous edema. Lateral collateral ligament complex intact.
3. Menisci: Complex tear of the posterior horn of the medial meniscus extending to the inferior articular surface. Lateral meniscus is normal in morphology and signal.
4. Compartments: Moderate tricompartmental osteoarthritis, most pronounced in medial joint space with severe cartilage loss. Mild patellofemoral chondrosis.
5. Soft Tissues / Joint: Moderate-to-large joint effusion with mild synovial thickening. Small Baker's cyst in popliteal fossa.
6. Bones: Bone contusion / marrow edema of the lateral tibial plateau. No acute cortical fracture identified.
```

### Mapping to the 12 Target Classes
| Target Name | Key Anatomical / Pathological Tokens | Positive Indicators | Negative / Normal Indicators |
|-------------|---------------------------------------|---------------------|------------------------------|
| `ACL` | acl, anterior cruciate | tear, disruption, rupture, retracted | intact, normal, unremarkable, continuous |
| `MCL` | mcl, medial collateral | sprain, tear, disruption, edema | intact, normal, unremarkable, continuous |
| `Medial Meniscus` | medial meniscus, medial meniscal, mm | tear, truncation, maceration, complex tear | intact, normal, unremarkable, no tear |
| `Lateral Meniscus` | lateral meniscus, lateral meniscal, lm | tear, fraying, truncation | intact, normal, unremarkable, no tear |
| `Medial OA` | medial compartment, medial joint, medial tibia/femur | osteoarthritis, cartilage loss, narrowing, osteophyte, wear | normal joint space, preserved cartilage |
| `Lateral OA` | lateral compartment, lateral joint | osteoarthritis, cartilage thinning, loss | preserved joint space, intact cartilage |
| `PF OA` | patellofemoral, trochlea, patella, pf joint | chondromalacia, osteoarthritis, wear, thinning | normal patellofemoral cartilage |
| `Effusion` | effusion, fluid | joint effusion, fluid collection, fluid accumulation | no effusion, physiological fluid, minimal |
| `Synovitis` | synovium, synovial | synovitis, synovial thickening, pannus | normal synovium, no synovitis |
| `Baker's` | baker, popliteal cyst | baker's cyst, popliteal cyst, fluid collection | no cyst, unremarkable popliteal fossa |
| `Contusion` | bone bruise, contusion, marrow edema | contusion, marrow edema, trabecular injury | no contusion, normal marrow signal |
| `Fracture` | fracture, cortical break | acute fracture, subchondral fracture, impacted | no fracture, intact cortex |

---

## 2. Negation & Certainty Scoping (Clinical NLP)

In medical narratives, simple keyword search fails because healthy structures are actively mentioned: *"The ACL is intact with no evidence of tear."* A naive search matches "ACL" and "tear", hallucinating an abnormality.

### NegEx Windowing & Syntactic Scoping
Negation terms operate within sentence boundaries:
- **Pre-negation**: `["no evidence of", "negative for", "without", "no", "absence of", "free of", "rules out"]`
- **Post-negation**: `["is intact", "is unremarkable", "is normal", "preserved", "shows no tear", "not identified"]`
- **Certainty Categories**:
  1. *Definitive Positive* (confidence $\approx 0.95$): "complete tear", "full thickness tear", "gross disruption", "marked osteoarthritis".
  2. *Probable / Moderate* (confidence $\approx 0.75$): "likely tear", "consistent with tear", "mild sprain", "moderate effusion".
  3. *Equivocal / Indeterminate* (confidence $\approx 0.50$): "cannot be excluded", "possible tear", "equivocal", "indeterminate".
  4. *Low Probability / Trace* (confidence $\approx 0.20$): "trace fluid", "mild irregularity without definite tear".
  5. *Definitive Negative* (confidence $\approx 0.05$): "intact", "unremarkable", "normal", "no tear", "no fracture".
  6. *Unmentioned / Default* (prior probability $\approx 0.10$): Structure omitted in abbreviated impression.

---

## 3. LLM Prompt Architecture & Zero-Shot Structured Output

When using an LLM (e.g., via local API or quantized medical SLM), we enforce strict JSON schema output:
```json
{
  "ACL": 0.95,
  "MCL": 0.20,
  "Medial Meniscus": 0.85,
  "Lateral Meniscus": 0.05,
  "Medial OA": 0.90,
  "Lateral OA": 0.10,
  "PF OA": 0.40,
  "Effusion": 0.95,
  "Synovitis": 0.70,
  "Baker's": 0.90,
  "Contusion": 0.80,
  "Fracture": 0.02
}
```
A structured prompt with definitions of all 12 classes ensures consistency and calibrated probability distribution.

---

## 4. Ground-Truth Alignment & Validation

### Alignment Evaluation
Using the subset of training data where ground truth annotations exist:
1. **Macro ROC AUC**: Verify report-extracted pseudo-probabilities against binary ground truth achieve $\text{Macro AUC} > 0.90$.
2. **Correlation**: Pearson and Spearman rank correlation $r > 0.80$.
3. **Reliability Diagram / Calibration**: Verify probabilities correlate with empirical positive rates.

### Silver Dataset Formulation
The generated `silver_train.csv` contains:
- `StudyInstanceUID`: Identifier matching DICOM slab caches.
- 12 Target Columns: Calibrated continuous pseudo-probabilities $[0.0, 1.0]$.
- `sample_weight`: Calculated from prediction certainty $w_i = \frac{1}{12}\sum_{c=1}^{12} 2 \cdot |p_{ic} - 0.5|$, ensuring high-certainty reports have higher gradient influence during vision model pretraining.
