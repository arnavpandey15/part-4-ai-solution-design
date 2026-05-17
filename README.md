# Part 4 – AI Solution Design for a Business Problem

> **Course**: Applied Neural Networks, Computer Vision, NLP, and AI Solution Design  
> **Role**: AI Business Analyst  
> **Domain**: Healthcare — AI-Assisted Chest X-Ray Triage  
> **Model**: DenseNet-121 Transfer Learning (CheXNet)

---

## Repository Structure

```
part-4-ai-solution-design/
├── README.md
├── notebook.ipynb                         ← All 8 tasks (fully documented)
├── requirements.txt
├── ai_usecase_reference_catalog.csv       ← Domain reference data
├── business_kpi_sample.csv               ← 12-month operational KPI data
└── results/
    ├── solution_summary.png               ← One-page visual blueprint (Task 8)
    ├── kpi_analysis.png                   ← KPI trends + AI impact projection
    ├── model_comparison_table.png         ← Model selection comparison table
    ├── model_comparison_table.csv         ← Machine-readable model comparison
    └── risk_matrix.png                    ← Responsible AI risk matrix (Task 7)
```

---

## Task 1 – Business Domain Selected

**Domain: Healthcare (Radiology)**

Selected from the AI Use Case Reference Catalog based on:
- Highest real-world impact (life-critical decisions)
- Direct relevance to course content (CNN/image classification)
- Richest responsible AI discussion surface
- Clear, measurable KPIs from the business_kpi_sample.csv dataset

---

## Task 2 – Business Problem Definition

### Problem: Unstructured Radiology Triage Causing Delayed Critical Diagnoses

| Dimension | Detail |
|---|---|
| **What problem is being solved?** | Chest X-rays arrive in chronological order. Critical findings (pneumothorax, mass lesions, consolidation) queue behind routine checkups. Radiologists review in arrival order, not clinical urgency. |
| **Who are the users/stakeholders?** | Radiologists, Emergency physicians, ICU teams, Hospital administrators, Patients |
| **Current manual process** | Sequential radiologist review. A critical X-ray arriving at 9am may not be reported until 3pm the next day (30+ hour average resolution time). |
| **Limitations of current process** | No automatic prioritisation · Human fatigue causes errors · Volume surges create backlogs · No scalability without hiring more radiologists |

### KPI Evidence (from `business_kpi_sample.csv`)

| KPI | Current Average | Problem Signal |
|---|---|---|
| Manual processing hours | 454 hrs / month | Radiologist time consumed by routine scans |
| Avg resolution time | 30.1 hours | Critical findings sit unread for over a day |
| Error rate | 6.9% | 1 in 14 scans misclassified or missed |
| CSAT score | 6.8 / 10 | Clinicians and patients dissatisfied with turnaround |
| Monthly scan volume | 2,778 scans | Growing +50 cases/month — unsustainable manually |

---

## Task 3 – AI Task Type

### Primary Task: Multi-Class Image Classification
### Secondary Task: Anomaly Detection

**Classification output**:
- `Normal` — routine scan, standard queue
- `Priority` — notable finding, elevated queue
- `Critical` — urgent pathology, immediate alert

**Why Image Classification?**

Chest X-rays are 2D spatial images. Pathological patterns that determine urgency (consolidations, pneumothorax, nodules) are **spatial, hierarchical visual features**:

```
Pixel intensities → Edges → Textures → Anatomical structures → Pathologies → Triage class
```

This is precisely the type of problem CNNs are architecturally designed to solve via learned convolutional filters at each level of the hierarchy.

**Why NOT other task types?**

| Task Type | Reason Not Selected |
|---|---|
| Text classification | Scan images cannot be processed as text |
| Regression | Triage requires categorical urgency, not a continuous score |
| Object detection | Localisation of findings is secondary; class label is primary |
| Anomaly detection (only) | Needs positive class definition for actionable triage output |

---

## Task 4 – Data Requirement Plan

| Dimension | Specification |
|---|---|
| **Type of data** | Unstructured: DICOM chest X-ray images (224×224×3) + Structured: patient metadata |
| **Input features** | Raw X-ray pixels, patient age, sex, referring department, prior scan history |
| **Target variable** | Triage label: `Normal` / `Priority` / `Critical` (labelled by 3 board-certified radiologists, majority vote) |
| **Data source** | Hospital PACS system + NIH ChestX-ray14 (112,120 frontal X-rays, open source) |
| **Collection method** | PACS integration via HL7 FHIR API; retrospective labelling from archived scan + radiology report pairs |
| **Minimum volume** | ≥ 10,000 labelled images per class (30,000 total) for effective fine-tuning |

### Data Quality Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Class imbalance (Critical cases ~5–8%) | High | Weighted loss function + class-balanced sampling |
| Scanner variation across hospital sites | High | Histogram normalisation per scanner type |
| Label noise (radiologist disagreement) | Medium | Majority vote from 3 radiologists; disputed cases excluded |
| Demographic bias (age, sex, ethnicity) | High | Stratified sampling; per-subgroup performance auditing |
| Missing patient metadata | Low | Train with metadata dropout — model works image-only |
| DICOM PHI leakage | Critical | Strip all metadata; train on raw pixel arrays only |

---

## Task 5 – Model Recommendation

### ★ Recommended: DenseNet-121 (CheXNet Transfer Learning)

| Model | Type | Est. AUC | Est. F1 | Recommended |
|---|---|---|---|---|
| Logistic Regression | Classical ML | 0.62 | 0.58 | ❌ Cannot process images |
| Vanilla CNN (from scratch) | CNN | 0.78 | 0.74 | ⚠️ Needs very large dataset |
| ResNet-50 (ImageNet) | Transfer Learning | 0.88 | 0.84 | ✅ Good general baseline |
| **DenseNet-121 (CheXNet) ★** | **Transfer Learning** | **0.95** | **0.91** | **✅✅ Best choice** |
| EfficientNet-B4 | Transfer Learning | 0.93 | 0.89 | ✅ Strong alternative |
| Vision Transformer (ViT) | Transformer | 0.96 | 0.93 | ⚠️ Needs large data, future option |

### Why DenseNet-121?

Stanford's CheXNet demonstrated DenseNet-121 achieves **radiologist-level performance** (AUC 0.841 on pneumonia detection — exceeding the average of 4 radiologists). Key DenseNet innovation: each layer receives feature maps from **all preceding layers**, maximising feature reuse and enabling gradients to flow directly to early layers — critical for detecting subtle medical texture differences.

### Architecture Pipeline

```
Input: 224×224×3 X-ray
    ↓ Initial Conv(7×7, 64 filters) + BatchNorm + ReLU + MaxPool
    ↓ Dense Block 1 (6 layers) → Transition Layer 1
    ↓ Dense Block 2 (12 layers) → Transition Layer 2
    ↓ Dense Block 3 (24 layers) → Transition Layer 3
    ↓ Dense Block 4 (16 layers)
    ↓ Global Average Pooling
    ↓ Dense(256, ReLU) → Dropout(0.5)
    ↓ Dense(3, Softmax) → [P(Normal), P(Priority), P(Critical)]
```

Fine-tuning strategy: Freeze DenseBlocks 1–3. Train DenseBlock 4 + classifier head for 10 epochs at lr=0.001. Unfreeze all layers, train full network at lr=0.0001 for 20 epochs.

---

## Task 6 – Evaluation Plan

### Technical Metrics

| Metric | Target | Rationale |
|---|---|---|
| Sensitivity — Critical class | ≥ 90% | Missing critical findings is catastrophically worse than a false alarm |
| Specificity | ≥ 85% | Avoid alert fatigue from too many false-positive Critical flags |
| AUC-ROC (macro OvR) | ≥ 0.95 | Discriminative power across all 3 classes |
| Macro F1-Score | ≥ 0.88 | Balanced precision/recall — equally penalises all class errors |
| Expected Calibration Error | ≤ 0.05 | Confidence scores must reliably reflect true probabilities |

### Business Metrics (vs. KPI Baseline)

| KPI | Current | AI Target | Measured Via |
|---|---|---|---|
| Resolution time | 30.1 hours | ≤ 6 hours | PACS timestamp: received → report signed |
| Manual hours/month | 454 hours | ≤ 140 hours | Radiologist time-tracking system |
| Error rate | 6.9% | ≤ 2.5% | Retrospective audit vs. pathology ground truth |
| CSAT score | 6.8 / 10 | ≥ 8.0 / 10 | Monthly clinician + patient satisfaction survey |

### Failure Cases & Safeguards

| Failure | Risk Level | Safeguard |
|---|---|---|
| Critical missed (False Negative) | Critical | Low threshold (0.40); mandatory radiologist review; SMS alert |
| Normal misclassified as Critical | Medium | Confidence display; feedback loop for calibration |
| Model confidence collapse | High | Automated AUC monitoring; fallback to manual queue |
| Distribution shift (new scanner) | Medium | Monthly performance audit; scanner-specific normalisation |

### Human Review Process

- All `Critical` predictions → immediate radiologist SMS + dashboard alert
- All confidence < 75% → senior radiologist urgent-review queue
- All predictions → logged (timestamp, scan ID, label, confidence, radiologist verdict)
- Monthly → performance audit comparing AI triage label vs. final diagnosis

---

## Task 7 – Responsible AI Considerations

### 1. Bias in Data
X-ray datasets historically over-represent US/European male patients. The model may under-detect pathologies in women, elderly, or underrepresented ethnic groups.  
**Mitigation**: Stratify training data by age, sex, ethnicity. Track per-subgroup AUC quarterly. Gap > 3% triggers mandatory retraining.

### 2. Incorrect Predictions (False Negatives)
A missed Critical classification leaves a life-threatening condition — pneumothorax, massive consolidation — unescalated. Clinical harm and legal liability follow.  
**Mitigation**: Conservative threshold (0.40) for Critical class. Mandatory radiologist review for all Critical flags. Sensitivity ≥ 90% is a hard deployment gate.

### 3. Patient Privacy
DICOM files contain PHI — patient name, date of birth, hospital ID. Training on raw DICOM creates HIPAA/GDPR exposure.  
**Mitigation**: Automated DICOM de-identification before training. Entirely on-premises deployment. Annual third-party privacy audit.

### 4. Over-Reliance / Automation Bias
Radiologists may progressively defer to AI triage labels without independent review, eroding clinical judgment over time.  
**Mitigation**: Dashboard shows confidence score, not just label. Radiologist final sign-off mandatory and non-waivable. Annual performance exercises without AI assistance.

### 5. Impact on Radiologists and Patients
Radiologists may perceive AI as surveillance. Patients may not know their scan was AI-triaged.  
**Mitigation**: Co-design with radiologists. AI positioned as "second reader". Patient consent forms updated; opt-out pathway available. Public performance metrics published.

### 6. Model Drift
Scanner upgrades, new disease patterns, or seasonal variation can shift the data distribution, silently degrading performance.  
**Mitigation**: Real-time monitoring dashboard. Automatic model pause if AUC drops > 2%. Quarterly retraining on new scanner data.

---

## Task 8 – Final Solution Summary

| Field | Detail |
|---|---|
| **Business Domain** | Healthcare (Radiology) |
| **Problem** | Unstructured X-ray triage: critical findings delayed 30+ hours |
| **Proposed AI Solution** | DenseNet-121 classifies each incoming X-ray as Normal/Priority/Critical, automatically reordering the radiologist work queue in real-time |
| **Required Data** | 30,000+ labelled chest X-rays (DICOM); patient metadata; radiologist labels (majority vote of 3) |
| **Model Recommendation** | DenseNet-121 (CheXNet pre-trained) fine-tuned on hospital data; Softmax(3) output |
| **Expected Business Impact** | Resolution time −80% · Error rate −65% · CSAT +25% · Manual hours −70% · ROI 3.8× Year 1 |
| **Key Risks** | Demographic bias · False negatives · Patient privacy · Radiologist over-reliance · Model drift |
| **Mitigation Plan** | Diverse training + subgroup audits · Low threshold + mandatory review · HIPAA de-identification · Confidence display + sign-off · Weekly monitoring |

---

## How to Run

```bash
git clone https://github.com/<your-username>/part-4-ai-solution-design
cd part-4-ai-solution-design
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Requirements

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
jupyter>=1.0
```