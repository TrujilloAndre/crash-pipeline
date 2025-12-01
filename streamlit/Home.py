# Home.py  — Streamlit "Home" page
# Run:  streamlit run Home.py
import os
import time
import requests
import streamlit as st
import base64
import duckdb


# --------------------------------------------
# Config
# --------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8080")  # your backend gateway (adjust as needed)
MINIO_URL   = os.getenv("MINIO_URL",   "http://minio:9000")
RABBIT_API  = os.getenv("RABBIT_API",  "http://rabbitmq:15672/api")
RABBIT_USER = os.getenv("RABBIT_USER", os.getenv("RABBIT_DEFAULT_USER","guest"))
RABBIT_PASS = os.getenv("RABBIT_PASS", os.getenv("RABBIT_DEFAULT_PASS","guest"))
GOLD_DB = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")
GOLD_TABLE = os.getenv("GOLD_TABLE", 'gold."gold"."crashes"')

# queues your workers consume
EXTRACT_Q   = os.getenv("EXTRACT_QUEUE", "extract")
TRANSFORM_Q = os.getenv("TRANSFORM_QUEUE", "transform")
CLEAN_Q     = os.getenv("CLEAN_QUEUE", "clean")

# If you want to link to other pages in the multi-page app, put the filenames here:
PAGE_LINKS = {
    "🧰 Data Management": "pages/Data_Management.py",
    "📡 Data Fetcher": "pages/Data_Fetcher.py",
    "⏰ Scheduler": "pages/Scheduler.py",
    "📊 EDA": "pages/EDA.py",
    "📑 Reports": "pages/Reports.py",
    "🧠 Model" : "pages/Model.py"
}

# Label overview content (edit freely to match your pipeline)
LABEL_CARDS = [
    {
        "emoji": "🚦",
        "name": "Crash Type (Injury/Tow)",
        "label_line": "Label: crash_type • Type: binary • Positive class: injury/tow (1)",
        "pipeline": "Model predicts injury/tow outcomes using context like lighting, surface, speed limit, and counts.",
        "features": [
            ("lighting_condition", "captures visibility context"),
            ("roadway_surface_cond", "signals traction risk"),
            ("posted_speed_limit", "proxy for severity potential"),
        ],
        "source_subset": {
            "crashes": ["crash_date", "weather_condition", "lighting_condition", "posted_speed_limit"],
            "vehicles": ["unit_type", "vehicle_year"],
            "people": ["person_type"],
        },
        "imbalance": "Positives: ~??% | Negatives: ~??% | Ratio: ~1:k (fill with QC stats)",
        "handling": "class_weight",
        "grain": "One row = crash",
        "window": "rolling per corr_id; latest transformed run",
        "filters": "drop null crash_record_id; standardize categories",
        "leakage": "dropped post-outcome fields, IDs not used as features",
        "gold_table": 'gold."gold"."crashes"',
    }
]

# --------------------------------------------
# Helpers
# --------------------------------------------

@st.cache_data(ttl=60)
def class_imbalance_from_duckdb(table: str, label_col: str, positive_value):
    con = duckdb.connect(GOLD_DB, read_only=True)
    # Works for numeric (0/1) or string labels (e.g., 'injury/tow')
    q = f"""
    WITH base AS (
      SELECT {label_col} AS y FROM {table}
      WHERE {label_col} IS NOT NULL
    )
    SELECT
      COUNT(*)::INTEGER AS n,
      SUM((y = $1))::INTEGER AS pos
    FROM base
    """
    df = con.execute(q, [positive_value]).fetchdf()
    n   = int(df.loc[0, "n"])
    pos = int(df.loc[0, "pos"])
    neg = n - pos
    pos_pct = (pos / n * 100.0) if n else 0.0
    neg_pct = (neg / n * 100.0) if n else 0.0
    ratio_k = round(neg / pos, 1) if pos > 0 else float("inf")
    return {
        "n": n, "pos": pos, "neg": neg,
        "pos_pct": pos_pct, "neg_pct": neg_pct, "ratio_k": ratio_k
    }

@st.cache_data(ttl=15)
def fetch_health():
    out = {"minio":"unknown","rabbitmq":"unknown","extractor":"unknown","transformer":"unknown","cleaner":"unknown"}

    # MinIO health
    try:
        r = requests.get(f"{MINIO_URL}/minio/health/ready", timeout=2)
        out["minio"] = "ok" if r.ok else "down"
    except Exception:
        out["minio"] = "down"

    # RabbitMQ overview + queue consumers
    try:
        auth = (RABBIT_USER, RABBIT_PASS)
        r = requests.get(f"{RABBIT_API}/overview", auth=auth, timeout=3)
        out["rabbitmq"] = "ok" if r.ok else "down"

        # helper to read consumer_count for a queue
        def q_health(qname):
            rq = requests.get(f"{RABBIT_API}/queues/%2F/{qname}", auth=auth, timeout=3)
            if not rq.ok:
                return "down"
            data = rq.json()
            return "ok" if data.get("consumers", 0) > 0 else "down"

        out["extractor"]   = q_health(EXTRACT_Q)
        out["transformer"] = q_health(TRANSFORM_Q)
        out["cleaner"]     = q_health(CLEAN_Q)
    except Exception:
        out["rabbitmq"] = out.get("rabbitmq","down")
        # leave worker statuses as-is (unknown/down)

    return out

def status_badge(state: str) -> str:
    colors = {
        "ok": "#16a34a",        # green-600
        "down": "#dc2626",      # red-600
        "unknown": "#9ca3af",   # gray-400
    }
    text = {
        "ok": "Running",
        "down": "Not Responding",
        "unknown": "Unknown",
    }[state if state in text else "unknown"] if (text := {"ok":"Running","down":"Not Responding","unknown":"Unknown"}) else "Unknown"  # noqa: E731
    color = colors.get(state, colors["unknown"])
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:999px;font-size:0.85rem;">{text}</span>'

def card(title_md: str, body_md: str):
    with st.container(border=True):
        if title_md:
            st.markdown(title_md)
        st.markdown(body_md)


# --------------------------------------------
# UI
# --------------------------------------------
st.set_page_config(page_title="Crash ML – Home", page_icon="🏠", layout="wide")
st.title("🏠 Home")
st.caption("ML label overviews and pipeline health at a glance.")

# Quick nav to other pages
with st.container():
    cols = st.columns(len(PAGE_LINKS))
    for (label, target), c in zip(PAGE_LINKS.items(), cols):
        with c:
            st.page_link(target, label=label, icon=None)

st.divider()

# A) Label Overview Cards
st.subheader("🎯 Label Overviews")
for spec in LABEL_CARDS:
    try:
        imb = class_imbalance_from_duckdb(GOLD_TABLE, "crash_type", 1)  # 1 = positive class
        LABEL_CARDS[0]["imbalance"] = (
            f"Positives: ~{imb['pos_pct']:.1f}% | "
            f"Negatives: ~{imb['neg_pct']:.1f}% | "
            f"Ratio: ~1:{imb['ratio_k']}"
        )
    except Exception as e:
    # Non-fatal: keep placeholder if DB/path not reachable
        st.info(f"Could not compute class imbalance yet: {e}")
    features_md = "\n".join([f"- **{f}** — {why}" for f, why in spec["features"]])
    src_md = []
    for k, cols in spec["source_subset"].items():
        if cols:
            src_md.append(f"- **{k}**: {', '.join(cols)}")
    src_md = "\n".join(src_md) if src_md else "- (no additional source columns)"
    body = f"""
**{spec['emoji']} {spec['name']}**  
{spec['label_line']}

**Pipeline**: {spec['pipeline']}

**Key features (why they help)**  
{features_md}

**Source columns (subset)**  
{src_md}

**Class imbalance**  
- {spec['imbalance']}  
- Handling: *{spec['handling']}*

**Data grain & filters**  
- {spec['grain']}  
- Window: {spec['window']}  
- Filters: {spec['filters']}

**Leakage/caveats**  
- {spec['leakage']}

**Gold table**: `{spec['gold_table']}`
"""
    card(title_md="", body_md=body)

st.divider()

# B) Container Health
st.subheader("🩺 Container Health")
health = fetch_health()

cols = st.columns(5)
services = ["minio", "rabbitmq", "extractor", "transformer", "cleaner"]
labels = ["MinIO", "RabbitMQ", "Extractor", "Transformer", "Cleaner"]

for c, key, label in zip(cols, services, labels):
    with c:
        state = health.get(key, "unknown")
        st.markdown(
            f"""
            <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fff;">
              <div style="font-weight:600;margin-bottom:6px;">{label}</div>
              {status_badge(state)}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Manual refresh
st.caption("Status auto-caches for 15s.")
if st.button("Refresh Health"):
    fetch_health.clear()
    st.rerun()

# Footer
st.write("")
st.caption("Backend stages: Extractor → Transformer → Cleaner → Gold (DuckDB). Use the tabs above to manage data and explore results.")
