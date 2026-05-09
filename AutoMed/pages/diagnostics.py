"""
pages/diagnostics.py — Window 2: Deep Diagnostics
===================================================
Displays:
  A. Executive clinical summary (classification + WHO treatment)
  B. AI explainability (SHAP waterfall + mutation impact table)
  C. Monitoring & safety flags
Depends on st.session_state populated by intro.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use non-interactive backend — critical for Streamlit
matplotlib.use("Agg")

# ── Gatekeeper ───────────────────────────────────────────────────────────────
if st.session_state.get("processed_data") is None:
    st.markdown("""
    <div class="tb-card tb-card-warn" style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧬</div>
        <div style="font-family: var(--font-display); font-size: 1.3rem; font-weight: 700;
                    margin-bottom: 0.5rem;">No Sequence Loaded</div>
        <div style="color: var(--text-muted); max-width: 380px; margin: 0 auto;">
            Please upload a FASTA file and run the pipeline on the
            <strong>Upload & Info</strong> page first.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Pull session state ────────────────────────────────────────────────────────
predictions    = st.session_state["predictions"]
report         = st.session_state["clinical_report"]
shap_values    = st.session_state["shap_values"]     # may be None in stub mode
processed_data = st.session_state["processed_data"]
patient_id     = st.session_state.get("patient_id", "ANON")
filename       = st.session_state.get("uploaded_filename", "unknown.fasta")

# ── Page header ───────────────────────────────────────────────────────────────
severity_class_map = {
    "🚨 CRITICAL": ("badge-critical", "CRITICAL"),
    "⚠️ WARNING":  ("badge-warning",  "WARNING"),
    "✅ STANDARD": ("badge-standard", "STANDARD"),
}
badge_cls, badge_label = severity_class_map.get(
    report.severity_flag, ("badge-standard", "STANDARD")
)

st.markdown(f"""
<div style="display: flex; align-items: flex-start; justify-content: space-between;
            margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem;">
    <div>
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-teal);
                    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.4rem;">
            Diagnostic Report · <span style="color: var(--text-muted);">{patient_id}</span>
        </div>
        <h1 style="font-size: 2rem; font-weight: 800; margin: 0; line-height: 1.1;">
            {report.tb_classification}
        </h1>
    </div>
    <div style="display: flex; gap: 0.6rem; align-items: center; padding-top: 0.2rem;">
        <span class="severity-badge {badge_cls}">{badge_label}</span>
        <span class="mono-tag">{filename}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SECTION A + B + C via Tabs ────────────────────────────────────────────────
tab_clinical, tab_ai, tab_monitoring = st.tabs([
    "💊  Clinical Recommendation",
    "🧠  AI Explanation",
    "📋  Monitoring & Safety",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB A — Clinical Recommendation
# ═══════════════════════════════════════════════════════════════════════
with tab_clinical:

    # ── High-visibility alert ──
    if report.classification_code in ("MDR", "RR"):
        st.error(
            f"{report.severity_flag}  —  **{report.tb_classification}**  |  "
            f"WHO 2024 Preferred Regimen: **{report.regimen_name}**"
        )
    elif report.classification_code == "HR":
        st.warning(
            f"{report.severity_flag}  —  **{report.tb_classification}**  |  "
            f"Recommended Regimen: **{report.regimen_name}**"
        )
    else:
        st.success(
            f"{report.severity_flag}  —  **{report.tb_classification}**  |  "
            f"Standard Regimen: **{report.regimen_name}**"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model predictions summary ──
    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                margin-bottom: 0.8rem; color: var(--text-primary);">
        Drug Susceptibility Testing Results
    </div>
    """, unsafe_allow_html=True)

    drug_cols = st.columns(len(predictions))
    drug_fullnames = {"RIF": "Rifampicin", "INH": "Isoniazid", "EMB": "Ethambutol", "FQ": "Fluoroquinolone"}

    for col, (drug, result) in zip(drug_cols, predictions.items()):
        label = result.get("label", "—")
        conf  = result.get("confidence", 0)
        is_r  = label == "R"
        color = "var(--accent-red)" if is_r else "var(--accent-teal)"
        status_text = "RESISTANT" if is_r else "SUSCEPTIBLE"

        col.markdown(f"""
        <div class="tb-card {'tb-card-critical' if is_r else 'tb-card-accent'}"
             style="text-align: center; padding: 1.2rem 0.8rem;">
            <div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted);
                        text-transform: uppercase; letter-spacing: 0.1em;">{drug}</div>
            <div style="font-family: var(--font-body); font-size: 0.7rem; color: var(--text-muted);
                        margin-bottom: 0.4rem;">{drug_fullnames.get(drug, drug)}</div>
            <div style="font-family: var(--font-display); font-size: 2.2rem; font-weight: 800;
                        color: {color}; line-height: 1;">{label}</div>
            <div style="font-family: var(--font-mono); font-size: 0.72rem; color: {color};
                        margin-top: 0.3rem; opacity: 0.8;">{status_text}</div>
            <div style="margin-top: 0.6rem;">
                <div style="background: var(--bg-elevated); border-radius: 4px;
                            height: 4px; overflow: hidden;">
                    <div style="width: {conf*100:.0f}%; height: 100%;
                                background: {color}; border-radius: 4px;"></div>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem;
                            color: var(--text-muted); margin-top: 4px;">
                    {conf:.1%} confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── WHO Treatment Table ──
    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                margin-bottom: 0.8rem;">
        WHO 2024 Treatment Recommendation
    </div>
    """, unsafe_allow_html=True)

    if report.bpal_downgrade:
        st.warning(f"🔄 **Regimen Downgrade:** {report.bpal_downgrade_reason}")

    treatment_df = pd.DataFrame({
        "Parameter": ["Regimen", "Drug Combination", "Duration", "Route"],
        "Detail": [
            report.regimen_name,
            report.regimen_full,
            f"{report.duration_months} months",
            "All-oral (no injections)",
        ]
    })
    st.table(treatment_df.set_index("Parameter"))

    st.info(f"📋 **Clinical Note:** {report.regimen_note}")

    # ── Tests required ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                margin-bottom: 0.6rem;">
        🔬 Additional Tests Required Before Treatment
    </div>
    """, unsafe_allow_html=True)

    tests_col1, tests_col2 = st.columns(2)
    tests = report.additional_tests_required
    mid = (len(tests) + 1) // 2
    for col, chunk in zip([tests_col1, tests_col2], [tests[:mid], tests[mid:]]):
        with col:
            for t in chunk:
                st.markdown(f"- {t}")

    # ── References ──
    with st.expander("📚 WHO Guidelines References"):
        for i, ref in enumerate(report.references, 1):
            st.markdown(f"{i}. {ref}")

    # ── Disclaimer ──
    st.divider()
    st.error(f"⚠️ **Clinical Disclaimer**  \n{report.disclaimer}")


# ═══════════════════════════════════════════════════════════════════════
# TAB B — AI Explanation
# ═══════════════════════════════════════════════════════════════════════
with tab_ai:

    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                margin-bottom: 0.3rem;">SHAP Feature Attribution</div>
    <p style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;
              max-width: 640px; margin-bottom: 1.5rem;">
        SHAP (SHapley Additive exPlanations) values show how each genomic feature
        pushes the model toward <span style="color: var(--accent-red); font-weight: 600;">
        Resistance (R)</span> or <span style="color: var(--accent-teal); font-weight: 600;">
        Susceptibility (S)</span>. The waterfall chart below breaks down the prediction
        for each drug individually.
    </p>
    """, unsafe_allow_html=True)

    # Drug selector
    drug_choice = st.selectbox(
        "Select drug to explain",
        options=list(predictions.keys()),
        format_func=lambda d: f"{d}  —  {drug_fullnames.get(d, d)}",
    )

    if shap_values is not None and drug_choice in shap_values:
        # ── Real SHAP waterfall (when your pipeline is connected) ──
        import shap
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#111720")
        ax.set_facecolor("#111720")
        shap.plots.waterfall(shap_values[drug_choice], show=False)
        st.pyplot(plt.gcf())
        plt.close()

    else:
        # ── Synthetic demo waterfall ── (remove when real SHAP values are connected)
        st.markdown("""
        <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--accent-amber);
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.8rem;">
            Demo visualization · Connect predictor.py for live SHAP values
        </div>
        """, unsafe_allow_html=True)

        # Build demo SHAP data per drug
        _demo_shap = {
            "RIF": {
                "features": ["rpoB_S450L", "rpoB_H445Y", "rpoB_D435V", "rpoB_Q432K",
                             "GC_content", "coverage_depth", "rpoB_other", "base_value"],
                "values":   [+4.21, +1.83, +0.44, +0.12, -0.38, -0.19, +0.07, 0.0],
                "baseline": 0.31,
                "output":   0.974,
            },
            "INH": {
                "features": ["katG_S315T", "inhA_C−15T", "katG_R463L", "ndh_R268H",
                             "GC_content", "coverage_depth", "katG_other", "base_value"],
                "values":   [+3.97, +0.62, +0.28, +0.09, -0.31, -0.14, +0.04, 0.0],
                "baseline": 0.28,
                "output":   0.941,
            },
            "EMB": {
                "features": ["embB_M306I", "embB_G406D", "embB_D328Y", "embC_other",
                             "GC_content", "coverage_depth", "embB_other", "base_value"],
                "values":   [-2.91, -0.74, -0.22, -0.11, +0.18, +0.09, -0.05, 0.0],
                "baseline": 0.55,
                "output":   0.118,
            },
            "FQ": {
                "features": ["gyrA_D94G", "gyrA_A90V", "gyrB_E501D", "gyrA_N538D",
                             "GC_content", "coverage_depth", "gyrA_other", "base_value"],
                "values":   [-3.12, -0.58, -0.18, -0.07, +0.14, +0.11, -0.03, 0.0],
                "baseline": 0.52,
                "output":   0.137,
            },
        }

        demo = _demo_shap.get(drug_choice, _demo_shap["RIF"])
        feats  = demo["features"][:-1]  # exclude base_value label
        vals   = demo["values"][:-1]
        base   = demo["baseline"]
        out    = demo["output"]
        label  = predictions[drug_choice]["label"]

        # Sort by absolute value descending
        order  = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
        feats  = [feats[i] for i in order]
        vals   = [vals[i]  for i in order]

        # Matplotlib waterfall
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#111720")
        ax.set_facecolor("#111720")

        colors = ["#ff4d6d" if v > 0 else "#00c9b1" for v in vals]
        y_pos  = np.arange(len(feats))

        # Horizontal bars
        cumulative = base
        bar_lefts  = []
        for i, v in enumerate(vals):
            bar_lefts.append(cumulative if v > 0 else cumulative + v)
            cumulative += v

        bars = ax.barh(
            y_pos, [abs(v) for v in vals],
            left=bar_lefts,
            color=colors,
            height=0.55,
            zorder=3,
        )

        # Value labels
        for i, (bar, v) in enumerate(zip(bars, vals)):
            x = bar.get_x() + bar.get_width() + 0.02 if v > 0 else bar.get_x() - 0.02
            ha = "left" if v > 0 else "right"
            ax.text(x, i, f"{'+' if v > 0 else ''}{v:.2f}",
                    va="center", ha=ha,
                    color="#ff4d6d" if v > 0 else "#00c9b1",
                    fontsize=8.5, fontfamily="monospace", fontweight="bold",
                    zorder=4)

        # Baseline + output lines
        ax.axvline(base, color="#3d4f6e", linewidth=1.2, linestyle="--", zorder=2,
                   label=f"Base value = {base:.3f}")
        ax.axvline(out,  color="#f0a500",  linewidth=1.8, linestyle="-",  zorder=5,
                   label=f"f(x) = {out:.3f}  [{label}]")

        # Axes styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontfamily="monospace", fontsize=9, color="#e8edf5")
        ax.set_xlabel("SHAP value (log-odds impact on Resistance probability)",
                      fontsize=8.5, color="#6b7a99", labelpad=8)
        ax.tick_params(colors="#6b7a99", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#1f2e45")
        ax.set_facecolor("#111720")
        ax.grid(axis="x", color="#1f2e45", linewidth=0.7, zorder=1)

        # Legend
        r_patch = mpatches.Patch(color="#ff4d6d", label="→ Resistance")
        s_patch = mpatches.Patch(color="#00c9b1", label="→ Susceptibility")
        ax.legend(
            handles=[r_patch, s_patch],
            loc="lower right",
            framealpha=0.0,
            labelcolor="#e8edf5",
            fontsize=8,
        )

        ax.set_title(
            f"SHAP Waterfall — {drug_choice}  ({drug_fullnames.get(drug_choice, drug_choice)})",
            fontsize=10.5, color="#e8edf5", fontweight="bold", pad=12,
            fontfamily="sans-serif",
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Mutation Impact Table ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                margin-bottom: 0.8rem;">
        Mutation Impact Table
    </div>
    """, unsafe_allow_html=True)

    # Mutation reference table with WHO grades and AI SHAP impact
    _mutation_db = {
        "RIF": [
            {"Mutation": "rpoB_S450L", "Gene": "rpoB", "Position": 450,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "+4.21 (Strong)"},
            {"Mutation": "rpoB_H445Y", "Gene": "rpoB", "Position": 445,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "+1.83 (Moderate)"},
            {"Mutation": "rpoB_D435V", "Gene": "rpoB", "Position": 435,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "+0.44 (Weak)"},
            {"Mutation": "rpoB_Q432K", "Gene": "rpoB", "Position": 432,
             "WHO Confidence": "Uncertain significance", "AI Impact (SHAP)": "+0.12 (Weak)"},
        ],
        "INH": [
            {"Mutation": "katG_S315T", "Gene": "katG", "Position": 315,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "+3.97 (Strong)"},
            {"Mutation": "inhA_C−15T", "Gene": "inhA", "Position": -15,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "+0.62 (Moderate)"},
            {"Mutation": "katG_R463L", "Gene": "katG", "Position": 463,
             "WHO Confidence": "Uncertain significance", "AI Impact (SHAP)": "+0.28 (Weak)"},
        ],
        "EMB": [
            {"Mutation": "embB_M306I", "Gene": "embB", "Position": 306,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "−2.91 (Strong ↓)"},
            {"Mutation": "embB_G406D", "Gene": "embB", "Position": 406,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "−0.74 (Moderate ↓)"},
        ],
        "FQ": [
            {"Mutation": "gyrA_D94G", "Gene": "gyrA", "Position": 94,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "−3.12 (Strong ↓)"},
            {"Mutation": "gyrA_A90V", "Gene": "gyrA", "Position": 90,
             "WHO Confidence": "Confirmed", "AI Impact (SHAP)": "−0.58 (Moderate ↓)"},
        ],
    }

    mut_data = _mutation_db.get(drug_choice, [])
    if mut_data:
        mut_df = pd.DataFrame(mut_data)
        st.dataframe(mut_df, use_container_width=True, hide_index=True)
    else:
        st.info("No mutation data available for this drug.")

    st.caption(
        "SHAP values are log-odds contributions. Positive = drives Resistance prediction; "
        "Negative = drives Susceptibility. WHO Confidence grades from WHO 2022 Mutation Catalogue."
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB C — Monitoring & Safety
# ═══════════════════════════════════════════════════════════════════════
with tab_monitoring:

    col_monitor, col_contra = st.columns([1, 1], gap="large")

    with col_monitor:
        st.markdown("""
        <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                    margin-bottom: 0.8rem;">📊 Monitoring Requirements</div>
        """, unsafe_allow_html=True)

        for item in report.monitoring_requirements:
            st.markdown(f"""
            <div class="tb-card" style="padding: 0.7rem 1rem; margin-bottom: 0.5rem;
                         display: flex; align-items: flex-start; gap: 0.7rem;">
                <span style="color: var(--accent-teal); font-size: 1rem;">›</span>
                <span style="font-size: 0.9rem; line-height: 1.5;">{item}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_contra:
        st.markdown("""
        <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                    margin-bottom: 0.8rem;">⚠️ Contraindication & Safety Flags</div>
        """, unsafe_allow_html=True)

        for flag in report.contraindication_flags:
            # Extract emoji prefix if present
            is_critical = "⚡" in flag or "🩸" in flag or "👁️" in flag
            border_class = "tb-card-warn" if is_critical else "tb-card"
            st.markdown(f"""
            <div class="tb-card {border_class}"
                 style="padding: 0.7rem 1rem; margin-bottom: 0.5rem; font-size: 0.88rem;
                        line-height: 1.55;">
                {flag}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Timeline view ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 1.05rem;
                margin-bottom: 1rem;">🗓️ Treatment Timeline</div>
    """, unsafe_allow_html=True)

    duration = report.duration_months
    timeline_items = {
        "DS": [
            ("Baseline", "DST, LFTs, CXR, HIV test"),
            ("Month 0–2", "Intensive phase: HRZE · Daily DOT"),
            ("Month 2", "Sputum culture · LFTs"),
            ("Month 2–6", "Continuation: HR · Monthly smear"),
            ("Month 6", "End of treatment · Culture confirmation"),
        ],
        "HR": [
            ("Baseline", "Confirmatory LPA, ECG, CXR, HIV/CD4"),
            ("Month 0–6", "REZLfx · Monthly sputum culture + LFTs"),
            ("Month 6", "End of treatment"),
        ],
        "MDR": [
            ("Baseline", "ECG, audiometry, CBC, LFTs, ophthalmology, HIV/CD4"),
            ("Month 1", "QTc · CBC · Sputum culture"),
            ("Month 1–6", "Monthly: ECG, CBC, sputum culture, neuropathy screen"),
            ("Month 3", "Thyroid function · Mid-treatment culture"),
            ("Month 6", "End of treatment · Final culture"),
        ],
        "RR": [
            ("Baseline", "Xpert Ultra confirm · ECG · Audiometry · HIV/CD4"),
            ("Month 1", "QTc · CBC · First sputum culture"),
            ("Month 1–6", "Monthly: ECG, CBC, sputum, neuropathy"),
            ("Month 6", "End of treatment · Two consecutive negative cultures"),
        ],
    }

    items = timeline_items.get(report.classification_code, timeline_items["DS"])
    cols_tl = st.columns(len(items))
    for col, (timepoint, action) in zip(cols_tl, items):
        col.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%;
                        background: var(--accent-teal); margin: 0 auto 0.5rem;
                        box-shadow: 0 0 8px var(--accent-teal);"></div>
            <div style="font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600;
                        color: var(--accent-amber); text-transform: uppercase;
                        letter-spacing: 0.06em; margin-bottom: 0.3rem;">{timepoint}</div>
            <div style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.5;">
                {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption(f"Generated: {report.generated_at}  ·  Patient: {patient_id}")
    st.error(f"⚠️ **Disclaimer**  \n{report.disclaimer}")