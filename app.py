
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Subtype Classifier",
    page_icon="🧬",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR = "Final_Models/"

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}

    # Gene list
    models["genes"] = joblib.load(MODEL_DIR + "gene_list.pkl")
    models["stream_indices"] = joblib.load(
        MODEL_DIR + "stream_indices.pkl")

    # Track A
    models["sc_a"]   = joblib.load(MODEL_DIR + "track_a_scaler.pkl")
    models["sel_a"]  = joblib.load(MODEL_DIR + "track_a_selector.pkl")
    models["enc_a"]  = joblib.load(MODEL_DIR + "track_a_encoder.pkl")
    models["imp_a"]  = joblib.load(MODEL_DIR + "track_a_imputer.pkl")
    models["lstm_a"] = tf.keras.models.load_model(
        MODEL_DIR + "track_a_lstm.keras")
    models["rf_full_a"] = joblib.load(
        MODEL_DIR + "track_a_rf_full.pkl")
    for name in ["Luminal","HER2","Basal","Proliferation",
                 "Immune","Full_Bio"]:
        models[f"rf_a_{name}"] = joblib.load(
            MODEL_DIR + f"track_a_rf_{name}.pkl")

    # Track B
    models["sc_b"]       = joblib.load(MODEL_DIR + "track_b_scaler.pkl")
    models["sel_b"]      = joblib.load(MODEL_DIR + "track_b_selector.pkl")
    models["enc_b"]      = joblib.load(MODEL_DIR + "track_b_encoder.pkl")
    models["imp_b"]      = joblib.load(MODEL_DIR + "track_b_imputer.pkl")
    models["imp_bio_b"]  = joblib.load(
        MODEL_DIR + "track_b_bio_imputer.pkl")
    models["lstm_b"]     = tf.keras.models.load_model(
        MODEL_DIR + "track_b_lstm.keras")
    models["rf_full_b"]  = joblib.load(
        MODEL_DIR + "track_b_rf_full.pkl")
    for name in ["Luminal","HER2","Basal","Proliferation",
                 "Immune","Full_Bio"]:
        models[f"rf_b_{name}"] = joblib.load(
            MODEL_DIR + f"track_b_rf_{name}.pkl")

    # Meta-learner + metadata
    models["meta"] = joblib.load(MODEL_DIR + "meta_learner.pkl")
    with open(MODEL_DIR + "model_metadata.json") as f:
        models["metadata"] = json.load(f)

    return models

# ── Block feature helpers ─────────────────────────────────────
def apply_block(X, sel, enc, imp, block_size=9):
    n_blocks = X.shape[1] // block_size
    usable   = n_blocks * block_size
    X_blk    = X[:, :usable].reshape(X.shape[0], n_blocks, block_size)
    stats    = np.hstack([X_blk.mean(axis=2), X_blk.std(axis=2),
                          X_blk.max(axis=2),  X_blk.min(axis=2)])
    return enc.transform(sel.transform(imp.transform(stats)))

# ── Inference ─────────────────────────────────────────────────
def predict(X_bio, track, models):
    """Run full GASE inference on one sample"""
    stream_idx = models["stream_indices"]

    if track == "A":
        sc, sel, enc, imp = (models["sc_a"], models["sel_a"],
                              models["enc_a"], models["imp_a"])
        rf_full  = models["rf_full_a"]
        lstm     = models["lstm_a"]
        rf_names = ["Luminal","HER2","Basal","Proliferation",
                    "Immune","Full_Bio"]
        rfs      = {n: models[f"rf_a_{n}"] for n in rf_names}
        width    = models["metadata"]["meta_feature_width"]
        pad_left = False
    else:
        imp_bio  = models["imp_bio_b"]
        X_bio    = imp_bio.transform(X_bio)
        sc, sel, enc, imp = (models["sc_b"], models["sel_b"],
                              models["enc_b"], models["imp_b"])
        rf_full  = models["rf_full_b"]
        lstm     = models["lstm_b"]
        rf_names = ["Luminal","HER2","Basal","Proliferation",
                    "Immune","Full_Bio"]
        rfs      = {n: models[f"rf_b_{n}"] for n in rf_names}
        width    = models["metadata"]["meta_feature_width"]
        pad_left = True

    # Build dual features
    X_blk    = apply_block(X_bio, sel, enc, imp)
    X_dual   = np.hstack([X_bio, X_blk])
    X_sc     = sc.transform(X_dual)

    # Collect base learner probas
    probas = []
    for name, s_idx in stream_idx.items():
        probas.append(rfs[name].predict_proba(X_sc[:, s_idx]))
    probas.append(rf_full.predict_proba(X_sc))

    # LSTM
    X_lstm = X_sc[:, :89].reshape(1, 89, 1)
    p_lstm = lstm.predict(X_lstm, verbose=0)
    if not np.isnan(p_lstm).any():
        probas.append(p_lstm)

    meta_feat = np.hstack(probas)
    pad_size  = width - meta_feat.shape[1]
    if pad_left:
        meta_feat = np.hstack([np.zeros((1, pad_size)), meta_feat])
    else:
        meta_feat = np.hstack([meta_feat, np.zeros((1, pad_size))])

    proba  = models["meta"].predict_proba(meta_feat)[0]
    pred   = np.argmax(proba)
    conf   = proba[pred]
    return pred, conf, proba

# ── Demo samples (real normalised values) ────────────────────
def get_demo_sample(subtype, genes):
    """Return archetypal gene expression for each subtype"""
    np.random.seed(42)
    vals = np.random.randn(len(genes)) * 0.3
    gene_idx = {g: i for i, g in enumerate(genes)}

    if subtype == "Luminal A":
        for g in ["ESR1","PGR","FOXA1","GATA3","BCL2","MLPH"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(2.0, 3.0)
        for g in ["KRT5","KRT14","ERBB2","MKI67"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(-2.0, -1.0)

    elif subtype == "HER2-Enriched":
        for g in ["ERBB2","GRB7","CCNE1","CDC20","FGFR4"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(2.5, 3.5)
        for g in ["ESR1","PGR","KRT5"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(-2.0, -1.0)

    elif subtype == "Triple Negative":
        for g in ["KRT5","KRT14","KRT17","FOXC1","EGFR","PHGDH"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(2.0, 3.0)
        for g in ["ESR1","PGR","ERBB2","FOXA1"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(-2.5, -1.5)

    elif subtype == "Luminal B":
        for g in ["ESR1","PGR","FOXA1","MKI67","CCNB1","TOP2A"]:
            if g in gene_idx:
                vals[gene_idx[g]] = np.random.uniform(1.5, 2.5)

    return vals.reshape(1, -1)

# ── UI ────────────────────────────────────────────────────────
st.title("🧬 Breast Cancer Subtype Classifier")
st.markdown("**Dual Platform GASE v3** — Gene-Aware Subtype Ensemble")
st.markdown(
    "Classifies breast cancer into 4 molecular subtypes using "
    "RNA-seq gene expression data. "
    "Validated across 4,512 patients — exceeds published SOTA."
)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
**Model:** Dual Platform GASE v3

**Performance:**
- TCGA: 93.12% ± 1.64%
- SCAN-B: 92.63% ± 0.74%
- p < 0.01 vs SOTA (88.79%)

**Subtypes classified:**
- 🔵 Luminal A
- 🟡 Luminal B
- 🟠 HER2-Enriched
- 🔴 Triple Negative (Basal)

**89 curated bio genes used**

**Developer:** Dr. A. Dannie Macrin
SIMATS Engineering, Chennai
""")

# Load models
with st.spinner("Loading models..."):
    models = load_models()
genes = models["genes"]

st.success(f"Models loaded — {len(genes)} genes, "
           f"4,512 training patients")
st.divider()

# Platform selection
st.subheader("Step 1 — Select RNA-seq Platform")
platform = st.radio(
    "Which platform was your data generated on?",
    ["TCGA / RSEM (HiSeqV2)", "SCAN-B / FPKM (GSE81538, GSE96058)"],
    help="RSEM: TCGA-style counts. FPKM: SCAN-B / GEO datasets."
)
track = "A" if "RSEM" in platform else "B"
st.info(f"Selected: Track {'A (TCGA/RSEM)' if track=='A' else 'B (SCAN-B/FPKM)'}")

st.divider()

# Input method
st.subheader("Step 2 — Input Gene Expression Data")
input_method = st.tabs(["📁 Upload CSV", "🎯 Demo Mode"])

X_input = None

with input_method[0]:
    st.markdown("""
    **CSV format required:**
    - One row per patient (or single row for one patient)
    - Column headers = gene names
    - Values = normalised expression (log2 + z-score)
    - Must contain at least the 89 bio genes
    """)

    uploaded = st.file_uploader(
        "Upload expression CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded, index_col=0)
        st.write(f"Uploaded: {df_up.shape[0]} patients, "
                 f"{df_up.shape[1]} genes")

        # Find bio genes
        found = [g for g in genes if g in df_up.columns]
        missing = [g for g in genes if g not in df_up.columns]
        st.write(f"Bio genes found: {len(found)}/89")
        if missing:
            st.warning(f"Missing genes (will be imputed): {missing}")

        # Build aligned matrix
        X_up = np.zeros((df_up.shape[0], len(genes)))
        for i, g in enumerate(genes):
            if g in df_up.columns:
                X_up[:, i] = df_up[g].values
            else:
                X_up[:, i] = np.nan
        X_input = X_up
        st.success("File ready for prediction ✅")

with input_method[1]:
    st.markdown("Load an archetypal sample for each subtype "
                "to see the classifier in action.")
    demo_subtype = st.selectbox(
        "Choose demo subtype:",
        ["Luminal A", "Luminal B", "HER2-Enriched", "Triple Negative"]
    )
    if st.button("Load Demo Sample", type="primary"):
        X_input = get_demo_sample(demo_subtype, genes)
        st.success(f"Demo sample loaded: {demo_subtype} ✅")
        st.caption(
            "Archetypal expression pattern — "
            "not a real patient sample"
        )

st.divider()

# Prediction
st.subheader("Step 3 — Predict Subtype")
subtype_names  = ["Luminal A", "Luminal B",
                   "HER2-Enriched", "Triple Negative"]
subtype_colors = ["#2196F3", "#FFC107", "#FF9800", "#F44336"]
subtype_emoji  = ["🔵", "🟡", "🟠", "🔴"]

if X_input is not None:
    if st.button("🔬 Run Prediction", type="primary"):
        with st.spinner("Running GASE pipeline..."):
            results = []
            for i in range(X_input.shape[0]):
                x = X_input[i:i+1, :]
                pred, conf, proba = predict(x, track, models)
                results.append({
                    "patient": i+1,
                    "subtype": subtype_names[pred],
                    "confidence": conf,
                    "proba": proba,
                    "uncertain": conf < 0.70
                })

        # Show results
        for r in results:
            col1, col2 = st.columns([1, 2])
            with col1:
                idx = subtype_names.index(r["subtype"])
                st.markdown(
                    f"### {subtype_emoji[idx]} {r['subtype']}")
                conf_pct = r["confidence"] * 100
                st.metric("Confidence", f"{conf_pct:.1f}%")
                if r["uncertain"]:
                    st.warning(
                        "⚠️ Low confidence — recommend "
                        "clinical review"
                    )
                else:
                    st.success("✅ High confidence prediction")

            with col2:
                st.markdown("**Probability across all subtypes:**")
                df_proba = pd.DataFrame({
                    "Subtype":     [f"{subtype_emoji[i]} "
                                    f"{subtype_names[i]}"
                                    for i in range(4)],
                    "Probability": [f"{p*100:.1f}%"
                                    for p in r["proba"]]
                })
                st.dataframe(df_proba, hide_index=True,
                             use_container_width=True)

            if X_input.shape[0] > 1:
                st.divider()

        # Batch summary
        if len(results) > 1:
            st.subheader("Batch Summary")
            df_summary = pd.DataFrame([{
                "Patient":    r["patient"],
                "Subtype":    r["subtype"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Flag":       "⚠️ Review" if r["uncertain"]
                               else "✅ Confident"
            } for r in results])
            st.dataframe(df_summary, hide_index=True,
                         use_container_width=True)
            n_unc = sum(r["uncertain"] for r in results)
            st.info(
                f"{n_unc}/{len(results)} patients flagged "
                f"for clinical review "
                f"({n_unc/len(results)*100:.1f}%)"
            )
else:
    st.info("Upload a CSV or load a demo sample to begin.")

st.divider()
st.caption(
    "Dual Platform GASE v3 · Dr. A. Dannie Macrin · "
    "SIMATS Engineering · 2026 · "
    "For research use only — not a clinical diagnostic tool"
)
