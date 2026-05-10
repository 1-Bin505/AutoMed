
# TB-DST · AI Diagnostic System

> **AI-assisted decision support for *M. tuberculosis* drug-resistance prediction.**  

---

## Overview

TB-DST is a Streamlit application that accepts *M. tuberculosis* whole-genome sequences (FASTA) and returns:

- Per-drug resistance predictions (R/S) for **RIF, INH, EMB, FQ**
- SHAP feature-attribution waterfall charts explaining each prediction
- A WHO 2024–compliant treatment recommendation (DS-TB → BPaLM/BPaL)
- Monitoring requirements, safety flags, and contraindication prompts

The pipeline connects five Python modules: a sequence processor, an ML inference engine, a SHAP explainability layer, a clinical decision module, and two Streamlit UI files.

---

## Repository Structure

```
.
├── app.py
|-- pages           # Single-page Streamlit entry point (recommended)
|    ├── intro.py          # Multi-page variant — Page 1: Upload & run
|    ├── diagnostics.py    # Multi-page variant — Page 2: Deep diagnostics
├── processor.py      # FASTA parsing and per-gene variant extraction
├── predictor.py      # ML model loader, resistance inference, SHAP computation
├── clinician.py      # WHO 2024 treatment classification and report generation
└── models/
    ├── RIF.pkl       # Trained XGBoost/sklearn model for Rifampicin
    ├── INH.pkl       # Trained model for Isoniazid
    └── EMB.pkl       # Trained model for Ethambutol
```
---

## Quick Start

### 1. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib scikit-learn xgboost shap
```

### 2. Add trained models

```
models/
├── RIF.pkl
├── INH.pkl
└── EMB.pkl
```

Each `.pkl` must contain either:

- A bare sklearn/XGBoost estimator with a `feature_names_in_` attribute, **or**
- A dict: `{"model": <estimator>, "feature_names": [str, ...]}`

### 3. Launch

```bash
# Single-page layout (recommended)
streamlit run app.py

# Multi-page layout
streamlit run intro.py
```

---

## Module Reference

### `processor.py` — Sequence Processor

Parses FASTA input and extracts binary mutation feature vectors per gene.

**Public API**

```python
from processor import process_fasta

result = process_fasta(file_content: str, gene_name: str) -> dict
# result = {
#   "success": bool,
#   "variants": [{"variant_name": str, ...}, ...]
# }
```

Called once per gene (`rpoB`, `katG`, `inhA`, `embA/B/C`, `gyrA`). Results are merged into a flat `{feature_name: 0|1}` dict that is passed downstream.

---

### `predictor.py` — Resistance Inference Engine

Loads trained `.pkl` models, aligns features, runs predictions, and computes SHAP explanations.

**Key class: `TBPredictor`**

```python
from predictor import TBPredictor

predictor = TBPredictor(
    model_paths=None,            # defaults to models/RIF.pkl, INH.pkl, EMB.pkl
    resistance_threshold=0.5,    # probability cut-off for "R" label
    top_n_shap=5,                # top SHAP contributors to surface
)

# Single-drug prediction
result = predictor.predict_resistance("RIF", mutation_vector)

# Multi-drug sweep (gene-aware routing)
results = predictor.run(mutation_vector)

# Summary DataFrame for UI rendering
df = predictor.summary_table(results)
```

**Result dict schema**

| Key | Type | Description |
|-----|------|-------------|
| `drug` | str | Drug abbreviation |
| `label` | str | `"R"` or `"S"` |
| `confidence` | float | P(Resistant) from model |
| `resistance_drivers` | list[dict] | Top SHAP features pushing toward R |
| `susceptibility_factors` | list[dict] | Top SHAP features pushing toward S |
| `all_contributions` | DataFrame | Full ranked SHAP table |
| `interpretation` | str | Human-readable confidence summary |

**Gene scope mapping** (controls feature routing to prevent cross-gene leakage)

| Drug | Genes |
|------|-------|
| RIF | `rpoB` |
| INH | `katG`, `inhA` |
| EMB | `embA`, `embB`, `embC` |

---

### `clinician.py` — Clinical Decision Module

Maps R/S labels to WHO 2024 TB classifications and treatment recommendations.

**Main entry point**

```python
from clinician import get_treatment_recommendation, ClinicalReport

report: ClinicalReport = get_treatment_recommendation(predictions)
```

**Classification logic**

| RIF | INH | Classification | Severity |
|-----|-----|---------------|----------|
| R | R | MDR-TB | 🚨 CRITICAL |
| R | S | RR-TB | 🚨 CRITICAL |
| S | R | Hr-TB | ⚠️ WARNING |
| S | S | DS-TB | ✅ STANDARD |

**BPaLM eligibility**

- BPaLM (Bedaquiline + Pretomanid + Linezolid + Moxifloxacin) is indicated for MDR-TB and RR-TB.
- Moxifloxacin is dropped (→ BPaL) if FQ resistance is detected, or if EMB resistance raises cross-resistance concern.

**`ClinicalReport` fields**

| Field | Description |
|-------|-------------|
| `tb_classification` | Full WHO classification string |
| `classification_code` | `"MDR"` / `"RR"` / `"HR"` / `"DS"` |
| `severity_flag` | `"🚨 CRITICAL"` / `"⚠️ WARNING"` / `"✅ STANDARD"` |
| `regimen_name` | Short regimen name (e.g. `"BPaLM"`) |
| `regimen_full` | Full drug expansion |
| `duration_months` | Treatment duration |
| `bpalm_eligible` | bool |
| `bpal_downgrade` | bool — true if Moxifloxacin excluded |
| `monitoring_requirements` | list[str] |
| `additional_tests_required` | list[str] |
| `contraindication_flags` | list[str] |

---

### `app.py` — Single-Page Streamlit UI

The recommended entry point. Handles upload, pipeline execution, results display, and SHAP visualisation in a single scrollable page.

**Session state keys written**

| Key | Type | Description |
|-----|------|-------------|
| `processed_data` | dict | Filename, sequence count, mutation vector |
| `predictions` | dict | Per-drug `{label, confidence}` |
| `shap_results` | dict \| None | Full predictor result dicts (with SHAP) |
| `clinical_report` | ClinicalReport | WHO recommendation |
| `patient_id` | str | From metadata form |
| `uploaded_filename` | str | Original filename |

**Metrics row** — all four metrics are derived live from `predictions` via `_compute_model_metrics()`. No values are hardcoded.

---

### `intro.py` / `diagnostics.py` — Multi-Page Variants

Alternative two-page layout. `intro.py` handles upload and pipeline execution; `diagnostics.py` renders the clinical and AI explanation tabs.

`diagnostics.py` reads SHAP data from `st.session_state["shap_results"]` (written by both `app.py` and the updated `intro.py`) with a fallback to the legacy `"shap_values"` key.

---

## Demo Mode

When no trained `.pkl` models are found in `models/`, the application falls back to **Demo Mode**:

- Predictions use a fixed illustrative stub (`_DEMO_STUB_PREDICTIONS`) with placeholder R/S labels and confidence values.
- The SHAP waterfall uses illustrative feature shapes scaled so their sum matches the live prediction confidence — the chart shape is representative but not from a real model.
- The mutation impact table shows the WHO 2022 Mutation Catalogue reference only; live SHAP values are omitted.
- All demo indicators are clearly labelled in the UI.

**Demo Mode is for interface preview only. It must not be used for any clinical purpose.**

---

## Data Flow

```
FASTA file
    │
    ▼
processor.py  ──► {feature: 0|1}  (mutation vector)
    │
    ▼
predictor.py  ──► {drug: {label, confidence, shap_values, ...}}
    │
    ├──► clinician.py  ──► ClinicalReport (WHO regimen, monitoring, flags)
    │
    └──► app.py / diagnostics.py  ──► Streamlit UI
```

---

## Supported Drugs & Genes

| Drug | Abbreviation | Key Resistance Genes | Key Mutations |
|------|-------------|---------------------|---------------|
| Rifampicin | RIF | `rpoB` | S450L, H445Y, D435V |
| Isoniazid | INH | `katG`, `inhA` | S315T (katG), C−15T (inhA) |
| Ethambutol | EMB | `embA`, `embB`, `embC` | M306I (embB) |
| Fluoroquinolone | FQ | `gyrA`, `gyrB` | D94G, A90V (gyrA) |

WHO Confidence grades sourced from the WHO 2022 Mutation Catalogue.

---

## Hardcoded Value Audit (Fixed)

The following issues were identified and resolved:

| File | Issue | Fix |
|------|-------|-----|
| `app.py` | Demo SHAP waterfall used fixed `baseline` and `output` values regardless of actual predictions | `base` fixed at 0.5; `out` sourced from live `predictions[drug]["confidence"]`; illustrative weights scaled proportionally |
| `app.py` | `STUB_PREDICTIONS` referenced directly in predictor fallback without clear labelling | Renamed to `_DEMO_STUB_PREDICTIONS` with a prominent comment; stubs only fill drugs not covered by loaded models |
| `intro.py` | Same `STUB_PREDICTIONS` issue; `_run_predictor` returned only predictions, discarding SHAP data | Renamed stub; function now returns `(predictions, shap_results)` tuple; `shap_results` stored in session state |
| `diagnostics.py` | Demo SHAP waterfall used hardcoded `baseline` and `output` per drug | Same fix as `app.py` — live confidence used for `out`, 0.5 for `base`, weights scaled |
| `diagnostics.py` | Mutation impact table showed hardcoded SHAP strings (e.g. `+4.21 (Strong)`) as if they were AI output | Column renamed to `Resistance Effect` with WHO catalogue text; live SHAP values merged in when a real model result is available |
| `diagnostics.py` | Read `st.session_state["shap_values"]` but `app.py` writes `"shap_results"` | Fixed to check `"shap_results"` first with fallback to `"shap_values"` |

---

## References

- WHO Consolidated Guidelines on Tuberculosis, Module 4 (2022, updated 2024)
- WHO Operational Handbook on Tuberculosis, Module 4 (2024)
- ZeNix Trial — Linezolid dose optimisation in BPaL (2021)
- TB-PRACTECAL Trial — BPaLM efficacy (2022)
- Nahid et al., ATS/CDC/ERS/IDSA Clinical Practice Guidelines for DS-TB (2016)
- WHO 2022 Mutation Catalogue for *M. tuberculosis* complex

---

## Disclaimer

> ⚠️ **AI-ASSISTED DECISION SUPPORT ONLY.**  
> This tool is generated by a machine learning model and must be reviewed and confirmed by a qualified clinician before any treatment is initiated. It is not a substitute for professional medical judgment, laboratory drug susceptibility testing (DST), or WHO-certified diagnostic methods.
