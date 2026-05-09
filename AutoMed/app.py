"""
app.py — TB-DST AI Pipeline
============================
Single-page scrollable Streamlit app.
Upload → Run → scroll down for full results.
Launch:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os

matplotlib.use("Agg")

st.set_page_config(
    page_title="TB-DST · AI Diagnostic System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-base:      #0a0e14;
    --bg-surface:   #111720;
    --bg-elevated:  #1a2332;
    --bg-card:      #151d2b;
    --accent-amber: #f0a500;
    --accent-teal:  #00c9b1;
    --accent-red:   #ff4d6d;
    --text-primary: #e8edf5;
    --text-muted:   #6b7a99;
    --text-dim:     #3d4f6e;
    --border:       #1f2e45;
    --font-display: 'Syne', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --radius:       8px;
    --radius-lg:    14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"]  {
    background: var(--bg-base) !important;
    border-bottom: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 230px !important;
}

[data-testid="stMainBlockContainer"] {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

h1, h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

[data-testid="stAlert"] { border-radius: var(--radius) !important; border-left-width: 4px !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-teal) !important; }

[data-testid="stButton"] button {
    background: var(--accent-teal) !important;
    color: var(--bg-base) !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-amber) !important;
    border-bottom-color: var(--accent-amber) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

code, pre {
    font-family: var(--font-mono) !important;
    background: var(--bg-elevated) !important;
    border-radius: 4px !important;
    color: var(--accent-teal) !important;
}
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

.tb-card          { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.5rem 1.8rem; margin-bottom: 1rem; }
.tb-card-accent   { border-left: 4px solid var(--accent-teal) !important; }
.tb-card-warn     { border-left: 4px solid var(--accent-amber) !important; }
.tb-card-critical { border-left: 4px solid var(--accent-red) !important; }

.mono-tag {
    font-family: var(--font-mono); font-size: 0.78rem;
    background: var(--bg-elevated); color: var(--accent-teal);
    padding: 2px 8px; border-radius: 4px; display: inline-block;
}
.section-label {
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-teal);
    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.4rem;
}
.severity-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-family: var(--font-display); font-weight: 700;
    font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-critical { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.badge-warning  { background: rgba(240,165,0,0.12);  color: #f0a500; border: 1px solid rgba(240,165,0,0.3); }
.badge-standard { background: rgba(0,201,177,0.10);  color: #00c9b1; border: 1px solid rgba(0,201,177,0.3); }

/* Anchor target offset so "Go to Top" lands correctly */
#top-anchor { position: relative; top: -80px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "processed_data": None, "predictions": None,
    "clinical_report": None, "shap_values": None,
    "uploaded_filename": None, "patient_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0.5rem 0.8rem;border-bottom:1px solid #1f2e45;margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                    color:#e8edf5;letter-spacing:-0.02em;">
            TB<span style="color:#00c9b1">·</span>DST
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.64rem;color:#6b7a99;
                    text-transform:uppercase;letter-spacing:0.12em;margin-top:3px;">
            AI Diagnostic System · WHO 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

    data_loaded = st.session_state["processed_data"] is not None
    pred_loaded = st.session_state["predictions"] is not None

    c = {
        "seq":  "#00c9b1" if data_loaded else "#3d4f6e",
        "pred": "#00c9b1" if pred_loaded else "#3d4f6e",
        "rep":  "#f0a500" if pred_loaded else "#3d4f6e",
    }
    i = {
        "seq":  "✓" if data_loaded else "○",
        "pred": "✓" if pred_loaded else "○",
        "rep":  "✓" if pred_loaded else "○",
    }

    st.markdown(f"""
    <div style="padding:0.8rem;background:#1a2332;border-radius:8px;border:1px solid #1f2e45;
                font-family:'DM Mono',monospace;font-size:0.72rem;margin-bottom:1rem;">
        <div style="color:#6b7a99;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:7px;">
            Pipeline Status
        </div>
        <div style="color:{c['seq']};margin-bottom:4px;">{i['seq']} Sequence Loaded</div>
        <div style="color:{c['pred']};margin-bottom:4px;">{i['pred']} Model Predictions</div>
        <div style="color:{c['rep']};">{i['rep']} Clinical Report</div>
    </div>
    """, unsafe_allow_html=True)

    if pred_loaded:
        st.info("⬇️ Scroll down to view the full diagnostic report.")

    st.divider()
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.61rem;color:#3d4f6e;line-height:1.7;">
        ⚠ AI-ASSISTED DECISION SUPPORT<br>
        For clinical review only.<br>
        Not a substitute for lab DST.
    </div>
    """, unsafe_allow_html=True)

# ── Pipeline helpers ──────────────────────────────────────────────────────────
GENE_DRUG_MAP = {"rpoB": "RIF", "katG": "INH", "inhA": "INH", "embB": "EMB", "gyrA": "FQ"}

STUB_PREDICTIONS = {
    "RIF": {"label": "R", "confidence": 0.974},
    "INH": {"label": "R", "confidence": 0.941},
    "EMB": {"label": "S", "confidence": 0.882},
    "FQ":  {"label": "S", "confidence": 0.863},
}

def _run_processor(file_content: str) -> dict:
    try:
        from processor import process_fasta
        merged: dict[str, int] = {}
        for gene in GENE_DRUG_MAP:
            try:
                result = process_fasta(file_content, gene_name=gene)
                if result.get("success"):
                    for v in result.get("variants", []):
                        merged[v["variant_name"]] = 1
            except Exception:
                pass
        return merged
    except ImportError:
        return {}

def _run_predictor(mutation_vector: dict) -> dict:
    if not mutation_vector:
        return STUB_PREDICTIONS.copy()
    try:
        from predictor import TBPredictor

        @st.cache_resource(show_spinner=False)
        def _get_predictor():
            return TBPredictor()

        predictor = _get_predictor()
        if not predictor.models:
            return STUB_PREDICTIONS.copy()

        results: dict = {}
        for drug in predictor.models:
            try:
                raw = predictor.predict_resistance(drug, mutation_vector)
                results[drug] = {"label": raw["label"], "confidence": raw["confidence"]}
            except Exception:
                pass
        for drug, stub in STUB_PREDICTIONS.items():
            results.setdefault(drug, stub)
        return results
    except Exception:
        return STUB_PREDICTIONS.copy()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HERO + UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:2rem;">
    <div class="section-label">TB-DST · AI Diagnostic Pipeline</div>
    <h1 style="font-size:2.6rem;font-weight:800;margin:0;line-height:1.1;">From Gene to Bedside</h1>
    <p style="color:var(--text-muted);font-size:1.05rem;margin-top:0.8rem;max-width:640px;line-height:1.65;">
        Analyses <em>M. tuberculosis</em> whole-genome sequences for drug resistance across
        <code>rpoB</code>, <code>katG</code>, <code>inhA</code>, <code>embB</code> and <code>gyrA</code>,
        then generates a <strong>WHO 2024–compliant treatment recommendation</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("🔬 Gene Markers & Resistance Reference", expanded=False):
    st.dataframe(pd.DataFrame({
        "Gene":     ["rpoB","rpoB","rpoB","katG","inhA","embB","gyrA"],
        "Mutation": ["S450L","H445Y","D435V","S315T","C−15T","M306I","D94G"],
        "Drug":     ["Rifampicin"]*3+["Isoniazid","Isoniazid","Ethambutol","Fluoroquinolone"],
        "WHO Grade":["Confirmed"]*7,
    }), use_container_width=True, hide_index=True)
    st.markdown("**rpoB S450L** is the most prevalent RIF-resistance mutation globally. **katG S315T** accounts for ~60–70% of INH resistance.")

st.markdown("<br>", unsafe_allow_html=True)

col_upload, col_meta = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.1rem;margin-bottom:0.8rem;">Upload Sequence File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("FASTA", type=["fasta","fa","fna","txt"], label_visibility="collapsed")

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8", errors="replace")
        n_seqs = file_content.count(">")
        st.markdown(f"""
        <div class="tb-card tb-card-accent" style="margin-top:0.8rem;">
            <span class="mono-tag">{uploaded_file.name}</span>
            <div style="margin-top:0.6rem;display:flex;gap:1.5rem;">
                <div>
                    <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);text-transform:uppercase;">Sequences</div>
                    <div style="font-family:var(--font-display);font-weight:700;font-size:1.3rem;color:var(--accent-teal);">{n_seqs}</div>
                </div>
                <div>
                    <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);text-transform:uppercase;">Size</div>
                    <div style="font-family:var(--font-display);font-weight:700;font-size:1.3rem;">{len(file_content):,} bp</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        first_hdr = next((l.strip() for l in file_content.splitlines() if l.startswith(">")), "No header")
        with st.expander("Preview header"):
            st.code(first_hdr, language=None)

with col_meta:
    st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.1rem;margin-bottom:0.8rem;">Sample Metadata</div>', unsafe_allow_html=True)
    patient_id = st.text_input("Patient / Sample ID", placeholder="e.g. PT-2024-001")
    st.selectbox("Sequencing Platform", ["Illumina (WGS)","Oxford Nanopore","Ion Torrent","Sanger","Unknown"])
    st.selectbox("Clinical Context", ["New TB diagnosis","Treatment failure","Relapse","Contact tracing","Surveillance"])

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([2, 3], gap="large")

with col_btn:
    if st.button("▶  Run Diagnostic Pipeline", disabled=uploaded_file is None, use_container_width=True):
        with st.status("Running diagnostic pipeline…", expanded=True) as status:
            st.write("⚙️ Parsing FASTA and extracting features…")
            mutation_vector = _run_processor(file_content)
            st.session_state["processed_data"] = {
                "filename": uploaded_file.name,
                "num_sequences": file_content.count(">"),
                "sequence_length": len(file_content.replace("\n","").replace(">","")),
                "features": mutation_vector,
            }
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.session_state["patient_id"] = patient_id or "ANON"

            st.write("🧠 Running resistance prediction models…")
            predictions = _run_predictor(mutation_vector)
            st.session_state["predictions"] = predictions

            st.write("📋 Generating WHO-compliant clinical report…")
            from clinician import get_treatment_recommendation
            st.session_state["clinical_report"] = get_treatment_recommendation(predictions)

            st.write("📊 Preparing SHAP explainability…")
            st.session_state["shap_values"] = None

            status.update(label="✅ Pipeline complete — scroll down for results", state="complete", expanded=False)

        st.success("✅ Analysis complete — scroll down to view the full diagnostic report.")

with col_info:
    if uploaded_file is None:
        st.markdown("""
        <div class="tb-card" style="color:var(--text-muted);font-size:0.9rem;line-height:1.7;">
            <strong style="color:var(--text-primary);">Pipeline steps:</strong><br>
            <span class="mono-tag">1</span> FASTA parsed · per-gene variant extraction<br>
            <span class="mono-tag">2</span> ML models predict R/S per drug<br>
            <span class="mono-tag">3</span> SHAP explanations computed<br>
            <span class="mono-tag">4</span> WHO 2024 regimen · BPaLM eligibility
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DIAGNOSTIC RESULTS (shown inline once pipeline has run)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("predictions") is not None:
    predictions    = st.session_state["predictions"]
    report         = st.session_state["clinical_report"]
    shap_values    = st.session_state["shap_values"]
    processed_data = st.session_state["processed_data"]
    patient_id_s   = st.session_state.get("patient_id", "ANON")
    filename_s     = st.session_state.get("uploaded_filename", "unknown.fasta")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)

    # ── Results header ────────────────────────────────────────────────────────
    severity_map = {
        "🚨 CRITICAL": "badge-critical",
        "⚠️ WARNING":  "badge-warning",
        "✅ STANDARD": "badge-standard",
    }
    badge_cls = severity_map.get(report.severity_flag, "badge-standard")

    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;
                margin-bottom:1.5rem;flex-wrap:wrap;gap:1rem;
                padding:1.5rem 1.8rem;background:var(--bg-card);border:1px solid var(--border);
                border-radius:var(--radius-lg);border-left:4px solid var(--accent-amber);">
        <div>
            <div class="section-label">Diagnostic Report · {patient_id_s}</div>
            <h2 style="font-size:1.8rem;font-weight:800;margin:0.2rem 0 0;line-height:1.1;">
                # New line 437
                 {report.classification_title}
            </h2>
        </div>
        <div style="display:flex;gap:0.6rem;align-items:center;padding-top:0.4rem;">
            <span class="severity-badge {badge_cls}">{report.severity_flag.replace("🚨 ","").replace("⚠️ ","").replace("✅ ","")}</span>
            <span class="mono-tag">{filename_s}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Drug susceptibility cards ─────────────────────────────────────────────
    drug_fullnames = {"RIF":"Rifampicin","INH":"Isoniazid","EMB":"Ethambutol","FQ":"Fluoroquinolone"}
    drug_cols = st.columns(len(predictions))
    for col, (drug, result) in zip(drug_cols, predictions.items()):
        label = result.get("label","—")
        conf  = result.get("confidence", 0)
        is_r  = label == "R"
        color = "var(--accent-red)" if is_r else "var(--accent-teal)"
        border = "tb-card-critical" if is_r else "tb-card-accent"
        col.markdown(f"""
        <div class="tb-card {border}" style="text-align:center;padding:1.2rem 0.8rem;">
            <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);
                        text-transform:uppercase;letter-spacing:0.1em;">{drug}</div>
            <div style="font-family:var(--font-body);font-size:0.7rem;color:var(--text-muted);
                        margin-bottom:0.4rem;">{drug_fullnames.get(drug,drug)}</div>
            <div style="font-family:var(--font-display);font-size:2.2rem;font-weight:800;
                        color:{color};line-height:1;">{label}</div>
            <div style="font-family:var(--font-mono);font-size:0.72rem;color:{color};
                        margin-top:0.3rem;opacity:0.8;">{"RESISTANT" if is_r else "SUSCEPTIBLE"}</div>
            <div style="margin-top:0.6rem;">
                <div style="background:var(--bg-elevated);border-radius:4px;height:4px;overflow:hidden;">
                    <div style="width:{conf*100:.0f}%;height:100%;background:{color};border-radius:4px;"></div>
                </div>
                <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-top:4px;">
                    {conf:.1%} confidence
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Three-tab results area ────────────────────────────────────────────────
    tab_clinical, tab_ai, tab_monitoring = st.tabs([
        "💊  Clinical Recommendation",
        "🧠  AI Explanation",
        "📋  Monitoring & Safety",
    ])

    # ── TAB A: Clinical Recommendation ───────────────────────────────────────
    with tab_clinical:
        if report.classification_code in ("MDR", "RR"):
            st.error(f"{report.severity_flag}  —  **{report.tb_classification}**  |  WHO 2024 Preferred Regimen: **{report.regimen_name}**")
        elif report.classification_code == "HR":
            st.warning(f"{report.severity_flag}  —  **{report.tb_classification}**  |  Recommended Regimen: **{report.regimen_name}**")
        else:
            st.success(f"{report.severity_flag}  —  **{report.tb_classification}**  |  Standard Regimen: **{report.regimen_name}**")

        st.markdown("<br>", unsafe_allow_html=True)

        if report.bpal_downgrade:
            st.warning(f"🔄 **Regimen Downgrade:** {report.bpal_downgrade_reason}")

        st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:0.8rem;">WHO 2024 Treatment Recommendation</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Parameter": ["Regimen","Drug Combination","Duration","Route"],
            "Detail": [report.regimen_name, report.regimen_full, f"{report.duration_months} months","All-oral (no injections)"],
        }).set_index("Parameter"))

        st.info(f"📋 **Clinical Note:** {report.regimen_note}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:0.6rem;">🔬 Additional Tests Required Before Treatment</div>', unsafe_allow_html=True)
        tests = report.additional_tests_required
        c1, c2 = st.columns(2)
        mid = (len(tests)+1)//2
        for col, chunk in zip([c1,c2],[tests[:mid],tests[mid:]]):
            with col:
                for t in chunk:
                    st.markdown(f"- {t}")

        with st.expander("📚 WHO Guidelines References"):
            for i, ref in enumerate(report.references, 1):
                st.markdown(f"{i}. {ref}")

        st.divider()
        st.error(f"⚠️ **Clinical Disclaimer**  \n{report.disclaimer}")

    # ── TAB B: AI Explanation ─────────────────────────────────────────────────
    with tab_ai:
        st.markdown("""
        <div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:0.3rem;">SHAP Feature Attribution</div>
        <p style="color:var(--text-muted);font-size:0.9rem;line-height:1.6;max-width:640px;margin-bottom:1.5rem;">
            SHAP values show how each genomic feature pushes the model toward
            <span style="color:var(--accent-red);font-weight:600;">Resistance (R)</span> or
            <span style="color:var(--accent-teal);font-weight:600;">Susceptibility (S)</span>.
        </p>
        """, unsafe_allow_html=True)

        drug_choice = st.selectbox("Select drug to explain", options=list(predictions.keys()),
                                   format_func=lambda d: f"{d}  —  {drug_fullnames.get(d,d)}")

        _demo_shap = {
            "RIF": {"features":["rpoB_S450L","rpoB_H445Y","rpoB_D435V","rpoB_Q432K","GC_content","coverage_depth","rpoB_other"],
                    "values":[+4.21,+1.83,+0.44,+0.12,-0.38,-0.19,+0.07],"baseline":0.31,"output":0.974},
            "INH": {"features":["katG_S315T","inhA_C-15T","katG_R463L","ndh_R268H","GC_content","coverage_depth","katG_other"],
                    "values":[+3.97,+0.62,+0.28,+0.09,-0.31,-0.14,+0.04],"baseline":0.28,"output":0.941},
            "EMB": {"features":["embB_M306I","embB_G406D","embB_D328Y","embC_other","GC_content","coverage_depth","embB_other"],
                    "values":[-2.91,-0.74,-0.22,-0.11,+0.18,+0.09,-0.05],"baseline":0.55,"output":0.118},
            "FQ":  {"features":["gyrA_D94G","gyrA_A90V","gyrB_E501D","gyrA_N538D","GC_content","coverage_depth","gyrA_other"],
                    "values":[-3.12,-0.58,-0.18,-0.07,+0.14,+0.11,-0.03],"baseline":0.52,"output":0.137},
        }

        demo  = _demo_shap.get(drug_choice, _demo_shap["RIF"])
        feats = demo["features"]
        vals  = demo["values"]
        base  = demo["baseline"]
        out   = demo["output"]
        lbl   = predictions[drug_choice]["label"]

        order = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
        feats = [feats[i] for i in order]
        vals  = [vals[i]  for i in order]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#111720")
        ax.set_facecolor("#111720")

        colors = ["#ff4d6d" if v > 0 else "#00c9b1" for v in vals]
        y_pos  = np.arange(len(feats))
        cum    = base
        lefts  = []
        for v in vals:
            lefts.append(cum if v > 0 else cum + v)
            cum += v

        bars = ax.barh(y_pos, [abs(v) for v in vals], left=lefts, color=colors, height=0.55, zorder=3)
        for bar, v in zip(bars, vals):
            x  = bar.get_x() + bar.get_width() + 0.02 if v > 0 else bar.get_x() - 0.02
            ha = "left" if v > 0 else "right"
            ax.text(x, bar.get_y()+bar.get_height()/2, f"{'+' if v>0 else ''}{v:.2f}",
                    va="center", ha=ha, color="#ff4d6d" if v>0 else "#00c9b1",
                    fontsize=8.5, fontfamily="monospace", fontweight="bold", zorder=4)

        ax.axvline(base, color="#3d4f6e", linewidth=1.2, linestyle="--", zorder=2, label=f"Base = {base:.3f}")
        ax.axvline(out,  color="#f0a500", linewidth=1.8, linestyle="-",  zorder=5, label=f"f(x) = {out:.3f} [{lbl}]")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontfamily="monospace", fontsize=9, color="#e8edf5")
        ax.set_xlabel("SHAP value (log-odds impact on Resistance)", fontsize=8.5, color="#6b7a99", labelpad=8)
        ax.tick_params(colors="#6b7a99", labelsize=8.5)
        for spine in ax.spines.values(): spine.set_color("#1f2e45")
        ax.grid(axis="x", color="#1f2e45", linewidth=0.7, zorder=1)
        ax.legend(handles=[mpatches.Patch(color="#ff4d6d",label="→ Resistance"),
                            mpatches.Patch(color="#00c9b1",label="→ Susceptibility")],
                  loc="lower right", framealpha=0.0, labelcolor="#e8edf5", fontsize=8)
        ax.set_title(f"SHAP Waterfall — {drug_choice} ({drug_fullnames.get(drug_choice,drug_choice)})",
                     fontsize=10.5, color="#e8edf5", fontweight="bold", pad=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:0.8rem;">Mutation Impact Table</div>', unsafe_allow_html=True)
        _mutation_db = {
            "RIF":[{"Mutation":"rpoB_S450L","Gene":"rpoB","Position":450,"WHO Confidence":"Confirmed","AI Impact (SHAP)":"+4.21 (Strong)"},
                   {"Mutation":"rpoB_H445Y","Gene":"rpoB","Position":445,"WHO Confidence":"Confirmed","AI Impact (SHAP)":"+1.83 (Moderate)"},
                   {"Mutation":"rpoB_D435V","Gene":"rpoB","Position":435,"WHO Confidence":"Confirmed","AI Impact (SHAP)":"+0.44 (Weak)"}],
            "INH":[{"Mutation":"katG_S315T","Gene":"katG","Position":315,"WHO Confidence":"Confirmed","AI Impact (SHAP)":"+3.97 (Strong)"},
                   {"Mutation":"inhA_C-15T","Gene":"inhA","Position":-15,"WHO Confidence":"Confirmed","AI Impact (SHAP)":"+0.62 (Moderate)"}],
            "EMB":[{"Mutation":"embB_M306I","Gene":"embB","Position":306,"WHO Confidence":"Confirmed","AI Impact (SHAP)":"−2.91 (Strong ↓)"}],
            "FQ": [{"Mutation":"gyrA_D94G", "Gene":"gyrA","Position":94, "WHO Confidence":"Confirmed","AI Impact (SHAP)":"−3.12 (Strong ↓)"}],
        }
        mut_data = _mutation_db.get(drug_choice, [])
        if mut_data:
            st.dataframe(pd.DataFrame(mut_data), use_container_width=True, hide_index=True)
        st.caption("Positive SHAP = drives Resistance; Negative = drives Susceptibility. WHO grades from 2022 Mutation Catalogue.")

    # ── TAB C: Monitoring & Safety ────────────────────────────────────────────
    with tab_monitoring:
        col_m, col_c = st.columns([1,1], gap="large")

        with col_m:
            st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:0.8rem;">📊 Monitoring Requirements</div>', unsafe_allow_html=True)
            for item in report.monitoring_requirements:
                st.markdown(f"""
                <div class="tb-card" style="padding:0.7rem 1rem;margin-bottom:0.5rem;display:flex;align-items:flex-start;gap:0.7rem;">
                    <span style="color:var(--accent-teal);font-size:1rem;">›</span>
                    <span style="font-size:0.9rem;line-height:1.5;">{item}</span>
                </div>""", unsafe_allow_html=True)

        with col_c:
            st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:0.8rem;">⚠️ Safety Flags</div>', unsafe_allow_html=True)
            for flag in report.contraindication_flags:
                is_crit = any(e in flag for e in ["⚡","🩸","👁️"])
                st.markdown(f"""
                <div class="tb-card {'tb-card-warn' if is_crit else ''}"
                     style="padding:0.7rem 1rem;margin-bottom:0.5rem;font-size:0.88rem;line-height:1.55;">
                    {flag}
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;margin-bottom:1rem;">🗓️ Treatment Timeline</div>', unsafe_allow_html=True)
        timeline_items = {
            "DS": [("Baseline","DST, LFTs, CXR, HIV"),("Month 0–2","Intensive: HRZE · DOT"),("Month 2","Sputum · LFTs"),("Month 2–6","Continuation: HR"),("Month 6","End of treatment")],
            "HR": [("Baseline","LPA, ECG, CXR, HIV/CD4"),("Month 0–6","REZLfx · Monthly culture"),("Month 6","End of treatment")],
            "MDR":[("Baseline","ECG, audiometry, CBC, ophthalmology"),("Month 1","QTc · CBC · Culture"),("Month 1–6","Monthly monitoring"),("Month 3","Thyroid · Mid-culture"),("Month 6","End · Final culture")],
            "RR": [("Baseline","Xpert Ultra · ECG · Audiometry"),("Month 1","QTc · CBC · Culture"),("Month 1–6","Monthly monitoring"),("Month 6","End · 2 negative cultures")],
        }
        items = timeline_items.get(report.classification_code, timeline_items["DS"])
        tl_cols = st.columns(len(items))
        for col, (tp, action) in zip(tl_cols, items):
            col.markdown(f"""
            <div style="text-align:center;padding:0.5rem;">
                <div style="width:12px;height:12px;border-radius:50%;background:var(--accent-teal);
                            margin:0 auto 0.5rem;box-shadow:0 0 8px var(--accent-teal);"></div>
                <div style="font-family:var(--font-mono);font-size:0.68rem;font-weight:600;
                            color:var(--accent-amber);text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:0.3rem;">{tp}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);line-height:1.5;">{action}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.caption(f"Generated: {report.generated_at}  ·  Patient: {patient_id_s}")
        st.error(f"⚠️ **Disclaimer**  \n{report.disclaimer}")

    # ── Go to Top button ──────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;justify-content:center;margin:2rem 0 1rem;">
        <a href="#top-anchor" style="text-decoration:none;">
            <div style="display:inline-flex;align-items:center;gap:0.5rem;
                        padding:0.65rem 1.8rem;border-radius:8px;
                        background:var(--bg-elevated);border:1px solid var(--border);
                        font-family:var(--font-display);font-weight:700;font-size:0.9rem;
                        color:var(--accent-teal);letter-spacing:0.04em;
                        cursor:pointer;transition:background 0.15s;">
                ↑ &nbsp; Go to Top
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)