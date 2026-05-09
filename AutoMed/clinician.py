"""
clinician.py — WHO-Compliant TB Clinical Decision Support Module
================================================================
Bridges AI model output (R/S labels from predictor.py) to bedside
treatment recommendations following WHO 2024 TB Treatment Guidelines.

⚠️  DISCLAIMER: This module is an AI-assisted decision support tool.
    All outputs are SUPPORTIVE only and must be reviewed by a qualified
    clinician before any treatment decision is made.

Author: TB-DST AI Pipeline
References:
  - WHO Consolidated Guidelines on Tuberculosis, Module 4 (2022, updated 2024)
  - WHO Operational Handbook on Tuberculosis (Module 4, 2024 update)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import datetime

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DrugResult:
    """Represents a single drug's resistance prediction."""
    label: str          # "R" (resistant) or "S" (susceptible)
    confidence: float   # 0.0 – 1.0
    drug: str           # Drug name, e.g. "RIF", "INH"

    def is_resistant(self) -> bool:
        return self.label.upper() == "R"

    def is_susceptible(self) -> bool:
        return self.label.upper() == "S"


@dataclass
class ClinicalReport:
    """Full clinical decision output produced by this module."""
    # Ensure these names match exactly what is used in get_treatment_recommendation
    classification_title: str  # e.g., "Multidrug-Resistant TB"
    classification_code: str   # e.g., "MDR", "RR", "DS"
    description: str           # The explanation text
    
    # These should have defaults so they are optional during initialization
    regimen_name: str = ""
    regimen_drugs: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    disclaimer: str = "AI-assisted decision support. Review with WHO 2024 guidelines."


# ---------------------------------------------------------------------------
# 1. TB Classification Flags
# ---------------------------------------------------------------------------

def classify_tb(predictions: dict) -> tuple[str, str, str]:
    """
    Map raw R/S labels to a WHO TB classification.

    Parameters
    ----------
    predictions : dict
        Output from predictor.py, structured as:
        {
            'RIF': {'label': 'R' or 'S', 'confidence': 0.97},
            'INH': {'label': 'R' or 'S', 'confidence': 0.91},
            'EMB': {'label': 'S', 'confidence': 0.88},   # optional
            'FQ':  {'label': 'S', 'confidence': 0.85},   # optional (Fluoroquinolone)
        }

    Returns
    -------
    (classification, code, severity_flag)
        e.g. ("MDR-TB (Multi-Drug Resistant)", "MDR", "🚨 CRITICAL")
    """
    rif = predictions.get("RIF", {}).get("label", "").upper()
    inh = predictions.get("INH", {}).get("label", "").upper()

    if rif == "R" and inh == "R":
        return (
            "MDR-TB (Multi-Drug Resistant Tuberculosis)",
            "MDR",
            "🚨 CRITICAL"
        )
    elif rif == "R" and inh != "R":
        return (
            "RR-TB (Rifampicin-Resistant Tuberculosis)",
            "RR",
            "🚨 CRITICAL"
        )
    elif rif != "R" and inh == "R":
        return (
            "Hr-TB (Isoniazid-Resistant Tuberculosis)",
            "HR",
            "⚠️ WARNING"
        )
    else:  # Both susceptible
        return (
            "DS-TB (Drug-Susceptible Tuberculosis)",
            "DS",
            "✅ STANDARD"
        )


# ---------------------------------------------------------------------------
# 2. BPaLM Eligibility Check
# ---------------------------------------------------------------------------

def check_bpalm_eligibility(predictions: dict, classification_code: str) -> tuple[bool, bool, str]:
    """
    Determine BPaLM eligibility and whether Moxifloxacin (M) must be dropped.

    WHO 2024: BPaLM is indicated for MDR-TB and RR-TB.
    The "M" is dropped (→ BPaL) if:
      - Fluoroquinolone resistance is detected (FQ model label = R), OR
      - EMB resistance is detected as a proxy for FQ cross-resistance risk
        (conservative flag; clinician should confirm).

    Parameters
    ----------
    predictions : dict
        Same structure as classify_tb() input.
    classification_code : str
        One of "MDR", "RR", "HR", "DS".

    Returns
    -------
    (bpalm_eligible, bpal_downgrade, downgrade_reason)
    """
    if classification_code not in ("MDR", "RR"):
        return False, False, ""

    fq_label  = predictions.get("FQ",  {}).get("label", "S").upper()
    emb_label = predictions.get("EMB", {}).get("label", "S").upper()

    if fq_label == "R":
        return True, True, (
            "Fluoroquinolone resistance detected (FQ model: R). "
            "Moxifloxacin dropped → regimen downgraded to BPaL."
        )
    elif emb_label == "R":
        return True, True, (
            "Ethambutol resistance detected (EMB model: R). "
            "Possible cross-resistance risk flagged by clinician review. "
            "Consider BPaL over BPaLM pending confirmatory DST."
        )
    else:
        return True, False, ""


# ---------------------------------------------------------------------------
# 3. Treatment Recommendation Engine
# ---------------------------------------------------------------------------

_REGIMENS = {
    "DS": {
        "name": "Standard HRZE",
        "full": "Isoniazid (H) + Rifampicin (R) + Pyrazinamide (Z) + Ethambutol (E)",
        "duration": 6,
        "note": (
            "Intensive phase: 2 months HRZE  |  Continuation phase: 4 months HR. "
            "DOT (directly observed therapy) recommended throughout."
        ),
    },
    "HR": {
        "name": "6-REZLfx",
        "full": "Rifampicin (R) + Ethambutol (E) + Pyrazinamide (Z) + Levofloxacin (Lfx)",
        "duration": 6,
        "note": (
            "WHO 2022 recommended regimen for Hr-TB. "
            "Levofloxacin replaces Isoniazid for the full 6 months."
        ),
    },
    "RR": {
        "name": "BPaLM",
        "full": "Bedaquiline (B) + Pretomanid (Pa) + Linezolid (L) + Moxifloxacin (M)",
        "duration": 6,
        "note": (
            "All-oral 6-month regimen. WHO 2024 preferred regimen for RR-TB. "
            "Cardiac monitoring (QTc) required for Bedaquiline + Moxifloxacin. "
            "Linezolid dose: 600 mg/day; consider 300 mg if toxicity emerges."
        ),
    },
    "MDR": {
        "name": "BPaLM",
        "full": "Bedaquiline (B) + Pretomanid (Pa) + Linezolid (L) + Moxifloxacin (M)",
        "duration": 6,
        "note": (
            "All-oral 6-month regimen. WHO 2024 preferred regimen for MDR-TB. "
            "Baseline and monthly ECG monitoring required. "
            "Audiometry at baseline due to Linezolid ototoxicity risk."
        ),
    },
}

_BPAL_OVERRIDE = {
    "name": "BPaL",
    "full": "Bedaquiline (B) + Pretomanid (Pa) + Linezolid (L)",
    "note": (
        "Moxifloxacin excluded due to fluoroquinolone resistance or cross-resistance concern. "
        "Duration extended to 6–9 months per clinician judgment. "
        "Consider ZeNix-dosing protocol for Linezolid (300 mg/day) to reduce toxicity."
    ),
}

_MONITORING = {
    "DS": [
        "Monthly sputum smear/culture during intensive phase",
        "LFTs at baseline and month 2",
        "Visual acuity if Ethambutol continued beyond 2 months",
    ],
    "HR": [
        "Monthly sputum smear/culture",
        "LFTs at baseline; monthly if pre-existing liver disease",
        "QTc interval at baseline (Levofloxacin has mild QT risk)",
    ],
    "RR": [
        "Baseline ECG + monthly QTc monitoring (Bedaquiline + Moxifloxacin)",
        "Monthly sputum culture until two consecutive negatives",
        "Full blood count monthly (Linezolid myelosuppression)",
        "Peripheral neuropathy screen monthly (Linezolid)",
        "Ophthalmology review at baseline and if visual symptoms arise",
        "Audiometry at baseline and monthly",
        "Serum lactate if lactic acidosis suspected",
    ],
    "MDR": [
        "Baseline ECG + monthly QTc monitoring (Bedaquiline + Moxifloxacin)",
        "Monthly sputum culture until two consecutive negatives",
        "Full blood count monthly (Linezolid myelosuppression)",
        "Peripheral neuropathy screen monthly",
        "Ophthalmology review at baseline and if visual symptoms arise",
        "Audiometry at baseline and monthly",
        "Thyroid function tests every 3 months (Linezolid)",
        "Serum lactate if lactic acidosis suspected",
    ],
}

_ADDITIONAL_TESTS = {
    "DS": ["Baseline CXR", "HIV test", "Sputum culture + DST confirmation"],
    "HR": [
        "Confirmatory LPA (line probe assay) for katG / inhA mutations",
        "Sputum culture + full DST panel",
        "HIV test + CD4 count",
        "Baseline CXR",
    ],
    "RR": [
        "Confirmatory Xpert MTB/RIF Ultra",
        "Sputum culture + comprehensive DST (including BDQ, LZD)",
        "Baseline audiometry",
        "Baseline ECG",
        "HIV test + CD4 count",
        "Baseline CXR + CT chest if available",
        "Fluoroquinolone DST before confirming BPaLM",
    ],
    "MDR": [
        "Confirmatory Xpert MTB/RIF Ultra",
        "Sputum culture + comprehensive DST (including BDQ, LZD, FQ)",
        "Baseline audiometry",
        "Baseline ECG",
        "HIV test + CD4 count",
        "Baseline CXR + CT chest if available",
    ],
}

_REFERENCES = [
    "WHO Consolidated Guidelines on Tuberculosis, Module 4 (2022)",
    "WHO Operational Handbook on TB Treatment, Module 4 (2024 update)",
    "ZeNix Trial — Linezolid dose optimization in BPaL (2021)",
    "TB-PRACTECAL Trial — BPaLM efficacy data (2022)",
    "Nahid et al., Official ATS/CDC/ERS/IDSA Clinical Practice Guidelines (2016) — DS-TB",
]


def get_treatment_recommendation(predictions: dict) -> ClinicalReport:
    # 1. Pull the actual labels from your ML model results
    # We use .get("label") because that's what predictor.py outputs
    rif_res = predictions.get("RIF", {}).get("label") == "R"
    inh_res = predictions.get("INH", {}).get("label") == "R"
    fq_res  = predictions.get("FQ", {}).get("label") == "R"

    # 2. Logic to determine classification
    if rif_res and inh_res:
        code, title = "MDR", "Multidrug-Resistant TB (MDR-TB)"
        desc = "Resistance to Rifampicin and Isoniazid detected via ML inference."
    elif rif_res:
        code, title = "RR", "Rifampicin-Resistant TB (RR-TB)"
        desc = "Rifampicin resistance detected. WHO recommends treating as MDR-TB."
    elif inh_res:
        code, title = "Hr", "Isoniazid-Resistant (Hr-TB)"
        desc = "Isoniazid resistance detected; Rifampicin remains susceptible."
    else:
        code, title = "DS", "Drug-Susceptible TB (DS-TB)"
        desc = "No resistance-conferring mutations detected by the model."

    # 3. Create the Report (This matches the dataclass above)
    report = ClinicalReport(
        classification_title=title,
        classification_code=code,
        description=desc
    )

    # 4. Assign Regimens dynamically
    if code in ["MDR", "RR"]:
        report.regimen_name = "BPaLM (6-month all-oral)"
        report.regimen_drugs = ["Bedaquiline", "Pretomanid", "Linezolid", "Moxifloxacin"]
    elif code == "Hr":
        report.regimen_name = "6-H-R-Z-Lfx"
        report.regimen_drugs = ["Rifampicin", "Ethambutol", "Pyrazinamide", "Levofloxacin"]
    else:
        report.regimen_name = "2HREZ / 4HR"
        report.regimen_drugs = ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide"]

    return report

# ---------------------------------------------------------------------------
# 4. Contraindication Flag Builder
# ---------------------------------------------------------------------------

def _build_contraindication_flags(predictions: dict, code: str) -> list[str]:
    """
    Surface key contraindication warnings based on the predicted regimen.
    These are prompts for the clinician to investigate — not automated blocks.
    """
    flags = []

    if code in ("MDR", "RR"):
        flags.append(
            "⚡ QTc Prolongation Risk: Bedaquiline + Moxifloxacin combination "
            "requires ECG monitoring. Avoid co-administration with other QT-prolonging agents."
        )
        flags.append(
            "🩸 Myelosuppression Risk: Linezolid may cause anaemia, thrombocytopenia. "
            "Baseline CBC required; monitor monthly."
        )
        flags.append(
            "👁️ Optic Neuropathy Risk: Linezolid associated with visual disturbance "
            "in prolonged use. Baseline ophthalmology assessment recommended."
        )

    if code == "HR":
        flags.append(
            "⚡ Mild QTc Risk: Levofloxacin has a low but non-zero QT-prolongation risk. "
            "Use caution in patients with pre-existing cardiac conditions."
        )

    # HIV co-infection note (always relevant for TB)
    flags.append(
        "🔴 HIV Co-infection: If patient is HIV-positive, review drug-drug interactions "
        "with ART. BDQ + ARTs (especially lopinavir/ritonavir) may affect QTc and BDQ levels."
    )

    return flags


# ---------------------------------------------------------------------------
# 5. Clinical Report Formatter (for Streamlit / text UI)
# ---------------------------------------------------------------------------

def format_report_text(report: ClinicalReport, patient_id: Optional[str] = None) -> str:
    """
    Render a ClinicalReport as a clean plain-text / markdown block
    suitable for Streamlit st.markdown() or st.text().

    Parameters
    ----------
    report : ClinicalReport
    patient_id : str, optional
        Anonymous patient identifier for audit trail.

    Returns
    -------
    str — formatted markdown string
    """
    pid_line = f"**Patient ID:** `{patient_id}`  \n" if patient_id else ""
    bpal_note = ""
    if report.bpal_downgrade:
        bpal_note = (
            f"\n> 🔄 **Regimen Downgrade:** {report.bpal_downgrade_reason}\n"
        )

    monitoring_lines = "\n".join(f"  - {m}" for m in report.monitoring_requirements)
    tests_lines      = "\n".join(f"  - {t}" for t in report.additional_tests_required)
    contra_lines     = "\n".join(f"  - {c}" for c in report.contraindication_flags)
    ref_lines        = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(report.references))

    return f"""
---
## {report.severity_flag} — WHO Clinical Decision Support Report
{pid_line}**Generated:** {report.generated_at}

---
### 🧬 TB Classification
**{report.tb_classification}**

---
### 💊 Recommended Regimen (WHO 2024)
| Field        | Detail |
|--------------|--------|
| **Regimen**  | {report.regimen_name} |
| **Drugs**    | {report.regimen_full} |
| **Duration** | {report.duration_months} months |

> 📋 **Note:** {report.regimen_note}
{bpal_note}
---
### 🔬 Additional Tests Required Before Treatment
{tests_lines}

---
### 📊 Monitoring Requirements
{monitoring_lines}

---
### ⚠️ Contraindication & Safety Flags
{contra_lines}

---
### 📚 References
{ref_lines}

---
> {report.disclaimer}
---
""".strip()


def format_report_dict(report: ClinicalReport) -> dict:
    """
    Return the ClinicalReport as a plain dict for JSON serialization
    or Streamlit metric/expander display.
    """
    return {
        "classification": report.tb_classification,
        "code": report.classification_code,
        "severity": report.severity_flag,
        "regimen": {
            "name": report.regimen_name,
            "drugs": report.regimen_full,
            "duration_months": report.duration_months,
            "note": report.regimen_note,
        },
        "bpalm": {
            "eligible": report.bpalm_eligible,
            "moxifloxacin_dropped": report.bpal_downgrade,
            "downgrade_reason": report.bpal_downgrade_reason,
        },
        "monitoring": report.monitoring_requirements,
        "tests_required": report.additional_tests_required,
        "contraindications": report.contraindication_flags,
        "references": report.references,
        "generated_at": report.generated_at,
        "disclaimer": report.disclaimer,
    }


# ---------------------------------------------------------------------------
# 6. Streamlit Integration Helper
# ---------------------------------------------------------------------------

def render_streamlit_report(report: ClinicalReport, patient_id: Optional[str] = None):
    """
    Renders the clinical report directly into a Streamlit app.
    Call this from your app.py / main Streamlit file.

    Requires: import streamlit as st
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("streamlit is required for render_streamlit_report()")

    # Severity colour mapping
    color_map = {
        "🚨 CRITICAL": "🔴",
        "⚠️ WARNING":  "🟡",
        "✅ STANDARD": "🟢",
    }
    dot = color_map.get(report.severity_flag, "⚪")

    st.divider()
    st.subheader(f"{dot} {report.tb_classification}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Regimen",   report.regimen_name)
    col2.metric("Duration",  f"{report.duration_months} months")
    col3.metric("WHO Status", report.classification_code)

    st.markdown(f"**Drugs:** {report.regimen_full}")
    st.info(f"📋 {report.regimen_note}")

    if report.bpal_downgrade:
        st.warning(f"🔄 **Regimen Downgrade:** {report.bpal_downgrade_reason}")

    with st.expander("🔬 Additional Tests Required"):
        for test in report.additional_tests_required:
            st.markdown(f"- {test}")

    with st.expander("📊 Monitoring Requirements"):
        for item in report.monitoring_requirements:
            st.markdown(f"- {item}")

    with st.expander("⚠️ Contraindication & Safety Flags"):
        for flag in report.contraindication_flags:
            st.markdown(f"- {flag}")

    with st.expander("📚 References"):
        for i, ref in enumerate(report.references, 1):
            st.markdown(f"{i}. {ref}")

    st.divider()
    st.caption(f"🕐 Generated: {report.generated_at}")
    st.error(report.disclaimer)


# ---------------------------------------------------------------------------
# 7. Quick Self-Test (run: python clinician.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        {
            "name": "DS-TB",
            "predictions": {
                "RIF": {"label": "S", "confidence": 0.98},
                "INH": {"label": "S", "confidence": 0.95},
            },
        },
        {
            "name": "Hr-TB",
            "predictions": {
                "RIF": {"label": "S", "confidence": 0.97},
                "INH": {"label": "R", "confidence": 0.92},
            },
        },
        {
            "name": "RR-TB",
            "predictions": {
                "RIF": {"label": "R", "confidence": 0.99},
                "INH": {"label": "S", "confidence": 0.89},
            },
        },
        {
            "name": "MDR-TB",
            "predictions": {
                "RIF": {"label": "R", "confidence": 0.98},
                "INH": {"label": "R", "confidence": 0.94},
            },
        },
        {
            "name": "MDR-TB + FQ resistance → BPaL",
            "predictions": {
                "RIF": {"label": "R", "confidence": 0.98},
                "INH": {"label": "R", "confidence": 0.94},
                "FQ":  {"label": "R", "confidence": 0.87},
            },
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {tc['name']}")
        print('='*60)
        report = get_treatment_recommendation(tc["predictions"])
        print(format_report_text(report, patient_id="PT-DEMO-001"))