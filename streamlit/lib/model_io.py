import os
import json
import joblib
import streamlit as st
import numpy as np


# Resolve paths via env (overridable), with sensible defaults
MODEL_ARTIFACT_PATH   = os.getenv("MODEL_ARTIFACT_PATH",   "/app/artifacts/model.pkl")
THRESHOLD_ARTIFACT    = os.getenv("THRESHOLD_PATH",        "/app/artifacts/threshold.txt")
LABELS_ARTIFACT       = os.getenv("LABELS_PATH",           "/app/artifacts/labels.json")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fitted sklearn Pipeline once per process."""
    try:
        pipe = joblib.load(MODEL_ARTIFACT_PATH)
        return pipe
    except Exception as e:
        # Bubble error in UI; pages can detect None
        st.error(f"Failed to load model from {MODEL_ARTIFACT_PATH}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_threshold(default_threshold: float = 0.5) -> float:
    """Load decision threshold for binary classifiers."""
    try:
        with open(THRESHOLD_ARTIFACT, "r") as f:
            return float(f.read().strip())
    except Exception:
        return default_threshold

@st.cache_resource(show_spinner=False)
def load_labels() -> dict:
    """Load label metadata (task type, positive class, etc.)."""
    try:
        with open(LABELS_ARTIFACT, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def positive_class_index(pipe, positive_class):
    """Find column index of the positive class in predict_proba output."""
    model = getattr(pipe, "named_steps", {}).get("model", None) or getattr(pipe, "steps", [[None, None]])[-1][1]
    classes_ = getattr(model, "classes_", None)
    if classes_ is None:
        return None
    idx = (classes_ == positive_class)
    where = np.where(idx)[0]
    return int(where[0]) if where.size > 0 else None
