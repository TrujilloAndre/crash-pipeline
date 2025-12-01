# streamlit/metrics.py
import time
import threading
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# -------------------------------
# Internal flag to avoid double-starting the HTTP server
# -------------------------------
_metrics_server_started = False
_metrics_lock = threading.Lock()

def ensure_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server on the given port, but only once.
    Safe to call from multiple Streamlit pages.
    """
    global _metrics_server_started
    if _metrics_server_started:
        return

    with _metrics_lock:
        if not _metrics_server_started:
            # Start /metrics endpoint on :port
            start_http_server(port)
            _metrics_server_started = True
            print(f"[metrics] Prometheus metrics server listening on :{port}/metrics")

# -------------------------------
# App / uptime
# -------------------------------
app_start_time = Gauge(
    "streamlit_app_start_time",
    "App startup timestamp (Unix epoch seconds)",
)
app_start_time.set(time.time())

# -------------------------------
# Model performance (latest eval)
# -------------------------------
model_accuracy = Gauge(
    "model_accuracy",
    "Latest evaluated model accuracy (from Streamlit Model page)",
)

model_last_trained_timestamp = Gauge(
    "model_last_trained_timestamp",
    "Unix timestamp (seconds) when the model was last evaluated/refreshed on the Model page",
)

model_training_duration_seconds = Gauge(
    "model_training_duration_seconds",
    "Duration in seconds of the last model evaluation run on the Model page",
)


model_precision = Gauge(
    "model_precision",
    "Latest evaluated model precision (from Streamlit Model page)",
)

model_recall = Gauge(
    "model_recall",
    "Latest evaluated model recall (from Streamlit Model page)",
)

# -------------------------------
# Prediction stats
# -------------------------------
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Latency of model predictions in seconds",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1, 2],
)

prediction_count = Counter(
    "prediction_total",
    "Total number of predictions made by the Streamlit app",
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction calls that raised exceptions",
)

data_rows_scored = Counter(
    "data_rows_scored_total",
    "Total number of data rows scored by the model",
)

# -------------------------------
# Helper functions
# -------------------------------
def update_model_metrics(accuracy: float, precision: float, recall: float) -> None:
    """
    Update the global model quality gauges so they can be read
    from the Monitoring page and scraped by Prometheus.
    """
    model_accuracy.set(float(accuracy))
    model_precision.set(float(precision))
    model_recall.set(float(recall))


def observe_prediction_latency(seconds: float) -> None:
    """
    Record a single prediction latency into the histogram.
    """
    prediction_latency.observe(float(seconds))

def update_training_timing(duration_seconds: float) -> None:
    """
    Record a 'training' event based on the Model page evaluation.
    - duration_seconds: how long the evaluation took
    """
    now = time.time()
    model_training_duration_seconds.set(float(duration_seconds))
    model_last_trained_timestamp.set(now)
