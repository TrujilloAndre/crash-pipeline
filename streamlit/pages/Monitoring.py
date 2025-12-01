# streamlit/pages/Monitoring.py
import time
import datetime as dt
import streamlit as st

from metrics import (
    model_accuracy,
    model_precision,
    model_recall,
    app_start_time,
    prediction_latency,
    prediction_count,
    prediction_errors,
    data_rows_scored,
)

st.title("📊 ML System Metrics Dashboard")

# -------------------------------
# Model quality (latest eval)
# -------------------------------
acc = model_accuracy._value.get()
prec = model_precision._value.get()
rec = model_recall._value.get()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", f"{acc:.3f}")
with col2:
    st.metric("Model Precision", f"{prec:.3f}")
with col3:
    st.metric("Model Recall", f"{rec:.3f}")

# -------------------------------
# App uptime
# -------------------------------
start_ts = app_start_time._value.get()
now_ts = time.time()
uptime_sec = max(0.0, now_ts - start_ts)

st.subheader("App Uptime")
st.write(f"Uptime: {uptime_sec / 60:.1f} minutes")
started_dt = dt.datetime.fromtimestamp(start_ts)
st.write("Started at:", started_dt.strftime("%Y-%m-%d %H:%M:%S"))

# -------------------------------
# Prediction latency summary (avg)
# -------------------------------
def read_histogram_from(hist):
    """
    Reads count & sum from a Histogram instance via its collected samples.
    """
    metric_family = next(iter(hist.collect()))
    total_count = 0
    total_sum = 0.0
    for sample in metric_family.samples:
        if sample.name.endswith("_count"):
            total_count = sample.value
        elif sample.name.endswith("_sum"):
            total_sum = sample.value
    return total_count, total_sum

count, total_sum = read_histogram_from(prediction_latency)
avg_latency = (total_sum / count) if count > 0 else 0.0

st.subheader("Prediction Performance")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Prediction Count", int(count))
with c2:
    st.metric("Avg Prediction Latency (sec)", f"{avg_latency:.4f}")
with c3:
    st.metric("Avg Latency (ms)", f"{avg_latency * 1000:.1f}")

# -------------------------------
# Extra counters (health signals)
# -------------------------------
st.subheader("Pipeline Health")

# These are Prometheus Counters; _value.get() gives current total
pred_total = prediction_count._value.get()
pred_errs = prediction_errors._value.get()
rows_scored = data_rows_scored._value.get()

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("Total Predictions", int(pred_total))
with c5:
    st.metric("Prediction Errors", int(pred_errs))
with c6:
    st.metric("Rows Scored", int(rows_scored))
