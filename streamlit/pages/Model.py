# streamlit/pages/Model.py
# Run as part of your multi-page app. Requires:
# - artifacts/model.pkl          (your fitted Pipeline / classifier)
# - artifacts/threshold.txt      (float; optional; defaults to 0.5)
# - artifacts/labels.json        (optional; not required for binary)
# - artifacts/metrics.json       (optional; “static” notebook metrics)
#
# Env expected (compose already sets these elsewhere in your app):
# - GOLD_DB_PATH  (e.g., /data/gold/gold.duckdb)
# - GOLD_TABLE    (e.g., gold."gold"."crashes")
# - MODEL_ARTIFACT_PATH (optional; defaults to /app/artifacts/model.pkl)
# - MODEL_THRESHOLD_PATH (optional; defaults to /app/artifacts/threshold.txt)

import os
import json
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import time

from joblib import load as joblib_load

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Prometheus / custom metrics
from metrics import (
    ensure_metrics_server,   # NEW
    prediction_count,
    prediction_errors,
    data_rows_scored,
    observe_prediction_latency,
    update_model_metrics,
    update_training_timing
)

# Start Prometheus metrics server on :8000 (safe to call multiple times)
ensure_metrics_server(port=8000)


# -------------------------------
# Config (env with sensible defaults)
# -------------------------------
GOLD_DB = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")
GOLD_TBL = os.getenv("GOLD_TABLE", 'gold."gold"."crashes"')

MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "/app/artifacts/model.pkl")
THRESHOLD_ARTIFACT = os.getenv("MODEL_THRESHOLD_PATH", "/app/artifacts/threshold.txt")
METRICS_JSON_PATH = os.getenv("MODEL_METRICS_PATH", "/app/artifacts/metrics.json")  # optional
POSITIVE_CLASS = int(os.getenv("POSITIVE_CLASS", "1"))  # binary positive class
DATE_COL_CANDIDATES = ["crash_date", "crash_datetime", "date_time"]  # used for filtering if present
ID_COL_CANDIDATES = ["crash_record_id", "record_id", "id"]

st.set_page_config(page_title="Model", page_icon="🧠", layout="wide")
st.title("🧠 Model")
st.caption("Load the trained model, select data, run predictions, and view metrics.")

# -------------------------------
# Cached helpers
# -------------------------------


@st.cache_resource(show_spinner=False)
def get_duckdb_conn(db_path: str):
    # read-only keeps us from accidentally mutating the warehouse via the UI
    return duckdb.connect(db_path, read_only=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    pipe = joblib_load(path)
    return pipe


@st.cache_data(show_spinner=False)
def load_threshold(path: str) -> float:
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 0.5  # default


@st.cache_data(show_spinner=False)
def load_static_metrics(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_existing(colnames, candidates):
    s = set(c.lower() for c in colnames)
    for c in candidates:
        if c.lower() in s:
            return c
    return None


@st.cache_data(show_spinner=False)
def query_gold_sample(
    db_path: str, table: str, start: str | None, end: str | None, limit: int = 5000
) -> pd.DataFrame:
    con = get_duckdb_conn(db_path)
    # detect a date col we can filter on
    cols = con.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf().columns.tolist()
    date_col = _pick_existing(cols, DATE_COL_CANDIDATES)

    if date_col and (start or end):
        where = []
        if start:
            where.append(f"{date_col} >= TIMESTAMP '{start} 00:00:00'")
        if end:
            where.append(f"{date_col} < TIMESTAMP '{end} 23:59:59'")
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        q = f"SELECT * FROM {table} {where_sql} LIMIT {int(limit)}"
    else:
        q = f"SELECT * FROM {table} LIMIT {int(limit)}"

    return con.execute(q).fetchdf()


def estimator_names(model_obj):
    """Return outer class and (if present) inner estimator class names."""
    outer = model_obj.__class__.__name__
    inner = None
    # If it’s a CalibratedClassifierCV or Pipeline, try to find underlying estimator
    if hasattr(model_obj, "estimator_"):
        inner = model_obj.estimator_.__class__.__name__
    elif hasattr(model_obj, "named_steps"):
        # it’s a Pipeline; try to guess final step name 'model'
        try:
            inner = model_obj.named_steps.get("model", list(model_obj.named_steps.values())[-1]).__class__.__name__
        except Exception:
            inner = None
    return outer, inner


def get_positive_index(clf, positive_value):
    """For binary classifiers with predict_proba, find column index for POSITIVE_CLASS."""
    try:
        classes_ = clf.classes_
        idx = np.where(classes_ == positive_value)[0]
        if len(idx) == 0:
            # fallback: assume positive is the max label (common for 0/1)
            idx = np.where(classes_ == classes_.max())[0]
        return int(idx[0])
    except Exception:
        return 1  # typical for [0,1]


def safe_predict_proba(pipe, X: pd.DataFrame, pos_value=1):
    # locate the classifier inside the pipeline to find the positive column
    clf = getattr(pipe, "named_steps", {}).get("model", pipe)
    idx = get_positive_index(clf, pos_value)
    probs = pipe.predict_proba(X)
    return probs[:, idx]


def expected_feature_names_from_pipeline(pipe):
    """
    Try to recover the raw input feature names the pipeline was fit on.
    Works if your preprocessor has feature_names_in_ (sklearn >= 1.0).
    """
    # Pipeline -> look for 'preproc' or first step with feature_names_in_
    if hasattr(pipe, "named_steps"):
        if "preproc" in pipe.named_steps and hasattr(pipe.named_steps["preproc"], "feature_names_in_"):
            return list(pipe.named_steps["preproc"].feature_names_in_)
        for step in pipe.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)
    return None  # unknown


# -------------------------------
# Section 1: Model Summary
# -------------------------------

with st.container(border=True):
    st.subheader("📦 Model Summary")

    # Load artifacts
    try:
        pipe = load_model(MODEL_ARTIFACT_PATH)
        threshold = load_threshold(THRESHOLD_ARTIFACT)
        outer, inner = estimator_names(pipe)
        st.success(f"Loaded model from `{MODEL_ARTIFACT_PATH}`", icon="✅")
        st.markdown(
            f"""
- **Outer object:** `{outer}`
- **Underlying estimator:** `{inner or 'N/A'}`
- **Decision threshold:** `{threshold:.2f}` (probability ≥ threshold → positive class = {POSITIVE_CLASS})
"""
        )
    except Exception as e:
        st.error(f"Failed to load model from `{MODEL_ARTIFACT_PATH}`: {e}")
        st.stop()

    # Feature expectations
    exp_feats = expected_feature_names_from_pipeline(pipe)
    if exp_feats:
        with st.expander("Expected raw feature columns (from training preprocessor)", expanded=False):
            st.code(", ".join(exp_feats), language="text")
        st.caption(
            "Note: One-hot encoding & numeric preprocessing happen **inside** the pipeline; "
            "pass raw columns with these names."
        )
    else:
        st.info(
            "Couldn’t infer expected feature names from the pipeline. "
            "The app will attempt to predict with the provided columns."
        )

    # Static metrics (optional)
    static_metrics = load_static_metrics(METRICS_JSON_PATH)
    if static_metrics:
        st.markdown("**Static metrics (from training notebook on held-out test):**")
        st.json(static_metrics, expanded=False)

# -------------------------------
# Section 2: Data Selection
# -------------------------------
st.subheader("🧮 Data Selection")

mode = st.radio(
    "Choose data source:",
    ["Gold table (sample / date filter)", "Upload test CSV"],
    horizontal=True,
)

data_df = None
with st.container(border=True):
    if mode == "Gold table (sample / date filter)":
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            max_rows = st.number_input("Max rows", min_value=100, max_value=100_000, value=5_000, step=500)
        with c2:
            start_date = st.date_input("Start date (optional)", value=None)  # None okay
        with c3:
            end_date = st.date_input("End date (optional)", value=None)

        start_s = start_date.isoformat() if isinstance(start_date, date) else None
        end_s = end_date.isoformat() if isinstance(end_date, date) else None

        if st.button("Load from Gold"):
            try:
                data_df = query_gold_sample(GOLD_DB, GOLD_TBL, start_s, end_s, max_rows)
                st.success(f"Loaded {len(data_df):,} rows from {GOLD_TBL}")
            except Exception as e:
                st.error(f"Query failed: {e}")

    else:  # Upload test CSV
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                data_df = pd.read_csv(up)
                st.success(f"Loaded {len(data_df):,} rows from uploaded file.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if data_df is not None:
        # Show a light preview
        st.caption("Preview (first 10 rows)")
        st.dataframe(data_df.head(10), use_container_width=True)

        # If we know expected features, check presence
        if exp_feats:
            missing = [c for c in exp_feats if c not in data_df.columns]
            extra = [c for c in data_df.columns if (c not in exp_feats and c != "crash_type")]
            if missing:
                st.warning(f"Missing expected columns ({len(missing)}). First few: {missing[:10]}")
            if extra:
                with st.expander("Extra columns present (ignored by the pipeline’s preprocessor)", expanded=False):
                    st.code(", ".join(extra), language="text")

# -------------------------------
# Section 3: Prediction & Metrics
# -------------------------------
st.subheader("📈 Prediction & Metrics")

if data_df is None:
    st.info("Load data above to enable predictions.")
    st.stop()

# 1) Build feature frame X (leave target if present for metrics)
target_col = "crash_type"  # your binary label (0/1)
has_target = target_col in data_df.columns

# Prefer the expected features if known; otherwise supply all columns (pipeline will select what it knows)
if exp_feats:
    X = data_df[[c for c in exp_feats if c in data_df.columns]].copy()
else:
    # Best-effort: pass everything but the explicit target (common when pipeline has ColumnTransformer with column lists)
    X = data_df.drop(columns=[target_col], errors="ignore").copy()

# 2) Predict probabilities (binary) and labels using stored threshold
try:
    t0 = time.perf_counter()
    proba_pos = safe_predict_proba(pipe, X, POSITIVE_CLASS)
    dt = time.perf_counter() - t0

    # Prometheus: record latency & counters
    observe_prediction_latency(dt)
    prediction_count.inc()
    data_rows_scored.inc(len(X))

    thr = load_threshold(THRESHOLD_ARTIFACT)
    y_pred = (proba_pos >= thr).astype(int)
except Exception as e:
    prediction_errors.inc()
    st.error(f"Prediction failed: {e}")
    st.stop()

# Attach predictions to a small preview (keep IDs if available)
id_col = _pick_existing(data_df.columns, ID_COL_CANDIDATES)
preview_cols = [c for c in [id_col, target_col] if c in data_df.columns]
pred_payload = {}
if id_col is not None:
    pred_payload[id_col] = data_df[id_col].values
pred_payload["proba_pos"] = proba_pos
pred_payload["pred"] = y_pred

pred_df = pd.DataFrame(pred_payload)
if target_col in data_df.columns:
    pred_df[target_col] = data_df[target_col].values

st.markdown("**Predictions (first 20)**")
st.dataframe(pred_df.head(20), use_container_width=True)

# 3) Live metrics (only if ground truth present)
if has_target:
    y_true = data_df[target_col].astype(int).values

    try:
        eval_start = time.perf_counter()

        # Compute live metrics
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, proba_pos)
        acc = accuracy_score(y_true, y_pred)

        eval_duration = time.perf_counter() - eval_start

        # Push into Prometheus Gauges so Monitoring/Grafana can see them
        update_model_metrics(acc, prec, rec)
        update_training_timing(eval_duration)

    except Exception as e:
        st.error(f"Metric computation failed: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 (live)", f"{f1:.3f}")
    c2.metric("Precision (live)", f"{prec:.3f}")
    c3.metric("Recall (live)", f"{rec:.3f}")
    c4.metric("ROC-AUC (live)", f"{auc:.3f}")

    with st.expander("Classification report (live)"):
        rep = classification_report(y_true, y_pred, digits=3)
        st.text(rep)

    with st.expander("Confusion matrix (live)"):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=False)
else:
    st.info("Ground truth column `crash_type` not found in data; showing predictions only (no live metrics).")
