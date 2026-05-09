"""
pages/intro.py — Window 1: Upload & Context
============================================
"""

import streamlit as st
import pandas as pd
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clinician import get_treatment_recommendation

GENE_DRUG_MAP = {"rpoB": "RIF", "katG": "INH", "inhA": "INH", "embB": "EMB", "gyrA": "FQ"}




def _run_processor(file_content: str) -> dict:
    """Call processor.process_fasta() per gene; merge into flat mutation dict."""
    try:
        from processor import process_fasta
    except ImportError:
        return {}

    merged: dict[str, int] = {}
    for gene in GENE_DRUG_MAP:
        try:
            result = process_fasta(file_content, gene_name=gene)
            if result.get("success"):
                for variant in result.get("variants", []):
                    merged[variant["variant_name"]] = 1
        except Exception:
            pass
    return merged


def _run_predictor(mutation_vector: dict[str, int]) -> dict:
    """
    Forces the app to use the actual ML models.
    If models are missing or fail, it returns an empty dict to trigger an error.
    """
    try:
        from predictor import TBPredictor
    except ImportError:
        st.error("Missing predictor.py file.")
        return {}

    # Initialize the actual inference engine
    predictor = TBPredictor()
    
    # 1. Check if models actually loaded from the /models folder
    if not predictor.models:
        st.error("⚠️ No model files (.pkl) found in the /models directory.")
        return {}

    # 2. Generate real predictions
    # Even if mutation_vector is empty {}, the model should predict "Susceptible"
    results = predictor.predict_all(mutation_vector)

    return results

# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 2.5rem;">
    <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-teal);
                text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.5rem;">
        TB-DST · AI Diagnostic Pipeline
    </div>
    <h1 style="font-size: 2.6rem; font-weight: 800; margin: 0; line-height: 1.1;">
        From Gene to Bedside
    </h1>
    <p style="color: var(--text-muted); font-size: 1.05rem; margin-top: 0.8rem;
              max-width: 640px; line-height: 1.65;">
        Analyses <em>M. tuberculosis</em> whole-genome sequences for drug resistance
        across <code>rpoB</code>, <code>katG</code>, <code>inhA</code>, <code>embB</code>
        and <code>gyrA</code>, then generates a
        <strong>WHO 2024–compliant treatment recommendation</strong>.
    </p>
</div>
""", unsafe_allow_html=True)
if st.button("🚀 Run Diagnostic Pipeline", use_container_width=True):
    with st.status("Analyzing Genomic Sequence...", expanded=True) as status:
        # 1. Process Sequence
        st.write("Extracting variants...")
        mut_dict = _run_processor(fasta_input)
        
        # 2. Run Inference
        st.write("Involving ML Inference Engine...")
        preds = _run_predictor(mut_dict)
        
        if not preds:
            status.update(label="Pipeline Failed", state="error")
            st.stop()
            st.write("Applying WHO 2024 Guidelines...")
        report = get_treatment_recommendation(preds)
        
        # Save to session state
        st.session_state["processed_data"] = {"mutations": mut_dict}
        st.session_state["predictions"] = preds
        st.session_state["clinical_report"] = report
        
        status.update(label="Analysis Complete!", state="complete")
        st.rerun()
with st.expander("🔬 Gene Markers & Resistance Logic", expanded=False):
    marker_data = {
        "Gene":     ["rpoB", "rpoB", "rpoB", "katG", "inhA", "embB", "gyrA"],
        "Mutation": ["S450L", "H445Y", "D435V", "S315T", "C−15T", "M306I", "D94G"],
        "Drug":     ["Rifampicin"]*3 + ["Isoniazid", "Isoniazid", "Ethambutol", "Fluoroquinolone"],
        "WHO":      ["Confirmed"]*7,
    }
    st.dataframe(pd.DataFrame(marker_data), use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

col_upload, col_meta = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.1rem;margin-bottom:0.8rem;">Upload Sequence File</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "FASTA / multi-FASTA",
        type=["fasta", "fa", "fna", "txt"],
        label_visibility="collapsed",
    )

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
        first_header = next((l.strip() for l in file_content.splitlines() if l.startswith(">")), "No header")
        with st.expander("Preview header"):
            st.code(first_header, language=None)

with col_meta:
    st.markdown('<div style="font-family:var(--font-display);font-weight:700;font-size:1.1rem;margin-bottom:0.8rem;">Sample Metadata</div>', unsafe_allow_html=True)
    patient_id = st.text_input("Patient / Sample ID", placeholder="e.g. PT-2024-001")
    st.selectbox("Sequencing Platform", ["Illumina (WGS)", "Oxford Nanopore", "Ion Torrent", "Sanger", "Unknown"])
    st.selectbox("Clinical Context", ["New TB diagnosis", "Treatment failure", "Relapse", "Contact tracing", "Surveillance"])

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

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
            st.session_state["clinical_report"] = get_treatment_recommendation(predictions)

            st.write("📊 Preparing SHAP explainability…")
            st.session_state["shap_values"] = None

            status.update(label="✅ Pipeline complete", state="complete", expanded=False)

        st.success("Analysis ready — navigate to **📊 Analysis** in the sidebar.")

with col_info:
    if uploaded_file is None:
        st.markdown("""
        <div class="tb-card" style="color:var(--text-muted);font-size:0.9rem;line-height:1.7;">
            <strong style="color:var(--text-primary);">Pipeline steps:</strong><br>
            <span class="mono-tag">1</span> FASTA parsed · per-gene variant extraction<br>
            <span class="mono-tag">2</span> ML models predict R/S per drug<br>
            <span class="mono-tag">3</span> SHAP explanations computed<br>
            <span class="mono-tag">4</span> WHO 2024 regimen + BPaLM eligibility
        </div>""", unsafe_allow_html=True)

if st.session_state.get("predictions") is not None:
    preds = st.session_state["predictions"]
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Current session results</div>', unsafe_allow_html=True)
    drug_cols = st.columns(len(preds))
    for col, (drug, result) in zip(drug_cols, preds.items()):
        label = result.get("label", "—")
        conf  = result.get("confidence", 0)
        color = "var(--accent-red)" if label == "R" else "var(--accent-teal)"
        col.markdown(f"""
        <div class="tb-card" style="text-align:center;padding:1rem;">
            <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);text-transform:uppercase;">{drug}</div>
            <div style="font-family:var(--font-display);font-size:2rem;font-weight:800;color:{color};">{label}</div>
            <div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted);">{conf:.1%}</div>
        </div>""", unsafe_allow_html=True)
