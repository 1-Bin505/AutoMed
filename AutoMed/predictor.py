"""
predictor.py — TB Resistance Inference Engine
==============================================
Bridges processor.py outputs → clinical SHAP-explained resistance predictions.

Supports: RIF (rpoB), INH (katG / inhA), EMB (embA/B/C)
"""

import pickle
import logging
import numpy as np
import pandas as pd
import shap
import streamlit as st
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drug → gene scope (must match training-time feature sets)
# ---------------------------------------------------------------------------
DRUG_GENE_SCOPE: dict[str, list[str]] = {
    "RIF": ["rpoB"],
    "INH": ["katG", "inhA"],
    "EMB": ["embA", "embB", "embC"],
}

# Default model paths (override via TBPredictor constructor)
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_PATHS: dict[str, Path] = {
    "RIF": DEFAULT_MODEL_DIR / "RIF.pkl",
    "INH": DEFAULT_MODEL_DIR / "INH.pkl",
    "EMB": DEFAULT_MODEL_DIR / "EMB.pkl",
}


# ---------------------------------------------------------------------------
# 1. Cached Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading resistance models…")
def load_models(model_paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    """
    Load and cache all drug-resistance .pkl files for the Streamlit session.

    Each .pkl must contain a dict with at least:
        {
            "model":         <sklearn/XGBoost estimator>,
            "feature_names": [str, ...]   # ordered list used at train time
        }

    Parameters
    ----------
    model_paths : dict[str, Path | str]
        Mapping of drug name → filesystem path to .pkl file.

    Returns
    -------
    dict[str, dict]
        drug → {"model": ..., "feature_names": [...]}
    """
    loaded: dict[str, dict[str, Any]] = {}
    for drug, path in model_paths.items():
        path = Path(path)
        if not path.exists():
            logger.warning("Model file not found for %s at %s — skipping.", drug, path)
            continue
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            # Normalise: accept bare estimator OR {"model": ..., "feature_names": ...}
            if isinstance(payload, dict) and "model" in payload:
                loaded[drug] = payload
            else:
                # Bare estimator — derive feature names if available
                feature_names = (
                    list(payload.feature_names_in_)
                    if hasattr(payload, "feature_names_in_")
                    else []
                )
                loaded[drug] = {"model": payload, "feature_names": feature_names}
            logger.info("Loaded model for %s (%d features).", drug,
                        len(loaded[drug]["feature_names"]))
        except Exception as exc:
            logger.error("Failed to load %s model: %s", drug, exc)
    return loaded


# ---------------------------------------------------------------------------
# 2. Probability Engine — feature-safe prediction
# ---------------------------------------------------------------------------
def _align_features(
    full_vector: dict[str, int],
    expected_features: list[str],
) -> np.ndarray:
    """
    Align the processor's full binary mutation dict to the model's expected
    feature order, filling missing features with 0.

    Parameters
    ----------
    full_vector : dict[str, int]
        {feature_name: 0|1} from processor.py
    expected_features : list[str]
        Feature names in the exact order the model was trained on.

    Returns
    -------
    np.ndarray shape (1, n_features)
    """
    aligned = np.array(
        [full_vector.get(feat, 0) for feat in expected_features],
        dtype=np.float32,
    ).reshape(1, -1)
    return aligned


def _filter_by_gene_scope(
    full_vector: dict[str, int],
    drug: str,
) -> dict[str, int]:
    """
    Retain only features whose prefix matches the drug's gene scope,
    preventing cross-gene leakage when the full mutation dict is passed in.
    """
    allowed_genes = DRUG_GENE_SCOPE.get(drug, [])
    if not allowed_genes:
        return full_vector  # no restriction defined
    return {
        feat: val
        for feat, val in full_vector.items()
        if any(feat.startswith(g) for g in allowed_genes)
    }


# ---------------------------------------------------------------------------
# 3. SHAP Explainability Layer
# ---------------------------------------------------------------------------
def _compute_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Calculate per-feature SHAP contributions and rank them.

    Parameters
    ----------
    model        : fitted sklearn / XGBoost / LightGBM tree estimator
    X            : np.ndarray shape (1, n_features) — single sample
    feature_names: list[str]
    top_n        : how many top drivers/factors to surface

    Returns
    -------
    dict with keys:
        shap_values       – raw array (n_features,)
        resistance_drivers – top_n positive-SHAP features (push toward R)
        susceptibility_factors – top_n negative-SHAP features (push toward S)
        all_contributions  – full sorted DataFrame
    """
    try:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)

        # Handle multi-output: take class-1 slice for binary classifiers
        if isinstance(raw, list):
            sv = raw[1][0]          # class "Resistant"
        elif raw.ndim == 3:
            sv = raw[0, :, 1]
        else:
            sv = raw[0]

    except Exception as exc:
        logger.warning("SHAP computation failed (%s); returning zeros.", exc)
        sv = np.zeros(len(feature_names))

    contrib_df = (
        pd.DataFrame({"feature": feature_names, "shap_value": sv})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    resistance_drivers = (
        contrib_df[contrib_df["shap_value"] > 0]
        .head(top_n)[["feature", "shap_value"]]
        .to_dict("records")
    )
    susceptibility_factors = (
        contrib_df[contrib_df["shap_value"] < 0]
        .head(top_n)[["feature", "shap_value"]]
        .to_dict("records")
    )

    return {
        "shap_values": sv,
        "resistance_drivers": resistance_drivers,
        "susceptibility_factors": susceptibility_factors,
        "all_contributions": contrib_df,
    }


# ---------------------------------------------------------------------------
# 4. Result Aggregator
# ---------------------------------------------------------------------------
def _build_result(
    drug: str,
    label: str,
    confidence: float,
    shap_data: dict,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Package prediction outputs into a clean, UI-ready dictionary.
    """
    return {
        "drug": drug,
        "label": label,                         # "R" | "S"
        "confidence": round(float(confidence), 4),
        "resistance_probability": round(float(confidence), 4),
        "shap_values": shap_data["shap_values"],
        "resistance_drivers": shap_data["resistance_drivers"],
        "susceptibility_factors": shap_data["susceptibility_factors"],
        "all_contributions": shap_data["all_contributions"],
        "feature_names": feature_names,
        "interpretation": (
            f"{'HIGH' if confidence >= 0.75 else 'MODERATE' if confidence >= 0.5 else 'LOW'} "
            f"probability of {drug} resistance ({confidence:.1%})"
        ),
    }


# ---------------------------------------------------------------------------
# 5. TBPredictor — main public class
# ---------------------------------------------------------------------------
class TBPredictor:
    """
    High-performance TB resistance inference engine.

    Usage
    -----
    predictor = TBPredictor()          # uses DEFAULT_MODEL_PATHS
    results   = predictor.run(mutation_vector)   # full multi-drug sweep

    Or single-drug:
    result = predictor.predict_resistance("RIF", mutation_vector)
    """

    def __init__(
        self,
        model_paths: dict[str, str | Path] | None = None,
        resistance_threshold: float = 0.5,
        top_n_shap: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        model_paths          : drug → path mapping (defaults to DEFAULT_MODEL_PATHS)
        resistance_threshold : probability cut-off for "R" label
        top_n_shap           : how many top SHAP contributors to surface
        """
        paths = model_paths or DEFAULT_MODEL_PATHS
        self.models: dict[str, dict] = load_models(
            {k: str(v) for k, v in paths.items()}
        )
        self.threshold = resistance_threshold
        self.top_n = top_n_shap

    # ------------------------------------------------------------------
    # A. Single-drug prediction
    # ------------------------------------------------------------------
    def predict_resistance(
        self,
        drug: str,
        mutation_vector: dict[str, int],
    ) -> dict[str, Any]:
        """
        Predict resistance for a single drug.

        Parameters
        ----------
        drug            : "RIF" | "INH" | "EMB"
        mutation_vector : {feature_name: 0|1} — full output of processor.py

        Returns
        -------
        Result dict (see _build_result for schema).
        """
        if drug not in self.models:
            raise ValueError(
                f"No model loaded for '{drug}'. "
                f"Available: {list(self.models.keys())}"
            )

        model_data = self.models[drug]
        model = model_data["model"]
        feature_names: list[str] = model_data["feature_names"]

        # --- Gene-scope filtering (EMB only sees emb genes, etc.)
        scoped_vector = _filter_by_gene_scope(mutation_vector, drug)

        # --- Feature alignment (handles mismatch gracefully)
        X = _align_features(scoped_vector, feature_names)

        # --- Probability Engine
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(proba[1])          # P(Resistant)
        except IndexError:
            confidence = float(model.predict_proba(X)[0][0])
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", drug, exc)
            raise

        label = "R" if confidence >= self.threshold else "S"

        # --- SHAP Layer
        shap_data = _compute_shap(model, X, feature_names, top_n=self.top_n)

        return _build_result(drug, label, confidence, shap_data, feature_names)

    # ------------------------------------------------------------------
    # B. Multi-drug sweep (gene-aware routing)
    # ------------------------------------------------------------------
    def run(
        self,
        mutation_vector: dict[str, int],
        drugs: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Intelligently route the mutation vector to each drug model.

        Gene-aware: only triggers a model if at least one mutation from its
        gene scope is present in mutation_vector (avoids running EMB when
        only rpoB mutations exist, etc.).

        Parameters
        ----------
        mutation_vector : full processor.py output
        drugs           : optional subset of drugs to evaluate
                          (defaults to all loaded models)

        Returns
        -------
        dict[drug, result_dict]
        """
        target_drugs = drugs or list(self.models.keys())
        results: dict[str, dict] = {}

        for drug in target_drugs:
            if drug not in self.models:
                logger.warning("Skipping %s — model not loaded.", drug)
                continue

            # --- Smart routing: only run if relevant mutations detected
            relevant_genes = DRUG_GENE_SCOPE.get(drug, [])
            if relevant_genes:
                has_relevant = any(
                    any(feat.startswith(g) for g in relevant_genes)
                    for feat in mutation_vector
                    if mutation_vector[feat] == 1
                )
                if not has_relevant:
                    logger.info(
                        "No %s-gene mutations found; skipping %s model.",
                        relevant_genes, drug,
                    )
                    continue

            try:
                results[drug] = self.predict_resistance(drug, mutation_vector)
                logger.info(
                    "%s → %s (%.1f%%)",
                    drug,
                    results[drug]["label"],
                    results[drug]["confidence"] * 100,
                )
            except Exception as exc:
                logger.error("Error predicting %s: %s", drug, exc)

        return results

    # ------------------------------------------------------------------
    # C. Convenience: summary table for Streamlit display
    # ------------------------------------------------------------------
    def summary_table(
        self, results: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Flatten multi-drug results into a summary DataFrame for UI rendering.

        Columns: Drug | Prediction | Confidence | Top Resistance Driver
        """
        rows = []
        for drug, r in results.items():
            top_driver = (
                r["resistance_drivers"][0]["feature"]
                if r["resistance_drivers"]
                else "—"
            )
            rows.append(
                {
                    "Drug": drug,
                    "Prediction": r["label"],
                    "Confidence": f"{r['confidence']:.1%}",
                    "Interpretation": r["interpretation"],
                    "Top Resistance Driver": top_driver,
                }
            )
        return pd.DataFrame(rows)