# pages/2_Data_Fetcher.py
# Streamlit "Data Fetcher" page: Streaming + Backfill publishers
import os
import json
import base64
from datetime import datetime, timezone, date, time as dtime

import requests
import streamlit as st
from minio import Minio
from minio.error import S3Error
import gzip, io, json as pyjson

# -----------------------------
# Config via environment
# -----------------------------
API_BASE       = os.getenv("API_BASE", "http://localhost:8080")  # optional backend for schema
RABBIT_API     = os.getenv("RABBIT_API", "http://rabbitmq:15672/api")
RABBIT_USER    = os.getenv("RABBIT_USER", os.getenv("RABBIT_DEFAULT_USER", "guest"))
RABBIT_PASS    = os.getenv("RABBIT_PASS", os.getenv("RABBIT_DEFAULT_PASS", "guest"))
EXTRACT_QUEUE  = os.getenv("EXTRACT_QUEUE", "extract")  # routing_key
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_USER", "minioadmin")
MINIO_PASS     = os.getenv("MINIO_PASS", "minioadmin")
MINIO_SSL      = os.getenv("MINIO_SSL", "false").strip().lower() in ("1","true","yes","on")
RAW_BUCKET     = os.getenv("RAW_BUCKET", "raw-data")           # where extractor writes raw JSON
RAW_PREFIX     = os.getenv("RAW_PREFIX", "crash")

# Optional dataset IDs (Socrata) if your extractor expects them
VEHICLES_DATASET_ID = os.getenv("VEHICLES_DATASET_ID", "68nd-jvt3")
PEOPLE_DATASET_ID   = os.getenv("PEOPLE_DATASET_ID", "u6pd-qa9d")


CRASHES_DATASET_ID = os.getenv("CRASHES_DATASET_ID", "85ca-t3if")

PRIMARY_SELECT = [
    "crash_record_id",
    "crash_date",
    "weather_condition",
    "lighting_condition",
    "posted_speed_limit",
    "injuries_total",
    "first_crash_type",
    "hit_and_run_i"
]


# -----------------------------
# Helpers
# -----------------------------
def _list_objects(cli: Minio, bucket: str, prefix: str, max_keys=50):
    seen = []
    for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
        if getattr(obj, "is_dir", False):
            continue
        seen.append(obj.object_name)
        if len(seen) >= max_keys:
            break
    return seen

def _read_json_array(cli: Minio, bucket: str, key: str):
    resp = None
    try:
        resp = cli.get_object(bucket, key)
        data = resp.read()
    finally:
        try:
            if resp: resp.close(); resp.release_conn()
        except Exception:
            pass
    # gunzip if needed
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
    try:
        arr = pyjson.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return []
    # normalize to list
    if isinstance(arr, list):
        return arr
    if isinstance(arr, dict) and isinstance(arr.get("data"), list):
        return arr["data"]
    return []

@st.cache_data(ttl=300)
def infer_columns_from_minio(dataset_alias: str) -> list[str]:
    """
    Look under raw-data/<RAW_PREFIX>/<alias>/... pick a few objects and union their keys.
    """
    try:
        cli = minio_client()
        # list objects under alias, prefer newest corr by sorting keys descending
        base = f"{RAW_PREFIX}/{dataset_alias}/"
        keys = _list_objects(cli, RAW_BUCKET, base, max_keys=200)
        # sort by key (assuming paths include corr/time); adjust if needed
        keys = sorted(keys, reverse=True)
        cols = set()
        took = 0
        for k in keys:
            if not (k.endswith(".json") or k.endswith(".json.gz")):
                continue
            rows = _read_json_array(cli, RAW_BUCKET, k)
            for row in rows[:50]:  # sample
                if isinstance(row, dict):
                    cols.update(map(str, row.keys()))
            took += 1
            if took >= 5 or len(cols) >= 200:
                break
        return sorted(cols)
    except S3Error as e:
        st.info(f"MinIO schema fallback failed for '{dataset_alias}': {e.code}")
    except Exception as e:
        st.info(f"MinIO schema fallback error for '{dataset_alias}': {e}")
    return []

def minio_client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=MINIO_SSL)

def _ts_corr() -> str:
    # corr_id like 2025-10-16-23-53-19
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

@st.cache_data(ttl=300)
def fetch_schema(dataset: str):
    # 1) try backend OpenAPI
    try:
        url = f"{API_BASE}/api/schema/{dataset}"
        r = requests.get(url, timeout=5)
        if r.ok:
            data = r.json()
            cols = data.get("columns") or data.get("fields") or []
            cols = [str(c) for c in cols]
            if cols:
                return cols
    except Exception:
        pass
    # 2) MinIO fallback
    alias = "vehicles" if dataset.lower().startswith("veh") else "people"
    cols = infer_columns_from_minio(alias)
    return cols  # [] if nothing found


def make_enrich(include_vehicles: bool, veh_cols: list,
                include_people: bool, ppl_cols: list):
    enrich = []
    if include_vehicles and veh_cols:
        enrich.append({
            "id": VEHICLES_DATASET_ID,
            "alias": "vehicles",
            "select": ",".join(veh_cols)
        })
    if include_people and ppl_cols:
        enrich.append({
            "id": PEOPLE_DATASET_ID,
            "alias": "people",
            "select": ",".join(ppl_cols)
        })
    return enrich

def publish_to_rabbit(payload: dict) -> tuple[bool, str]:
    """
    Publish via RabbitMQ HTTP API to default exchange -> routing_key=EXTRACT_QUEUE.
    """
    try:
        body_str = json.dumps(payload, separators=(",", ":"))
        body_b64 = base64.b64encode(body_str.encode("utf-8")).decode("ascii")
        api_url = f"{RABBIT_API}/exchanges/%2F/amq.default/publish"
        req = {
            "properties": {"content_type": "application/json", "delivery_mode": 2},
            "routing_key": EXTRACT_QUEUE,
            "payload": body_b64,
            "payload_encoding": "base64",
        }
        r = requests.post(api_url, auth=(RABBIT_USER, RABBIT_PASS),
                          headers={"content-type": "application/json"},
                          data=json.dumps(req),
                          timeout=8)
        if not r.ok:
            return False, f"HTTP {r.status_code}: {r.text}"
        resp = r.json()
        if resp.get("routed"):
            return True, "Queued"
        return False, f"Not routed: {resp}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def reset_form(keys: list[str]):
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Data Fetcher", page_icon="📡", layout="wide")
st.title("📡 Data Fetcher")
st.caption("Publish **Streaming** or **Backfill** fetch jobs with dynamic enrichment columns.")

tabs = st.tabs(["Streaming", "Backfill"])

# Load dynamic schema (cached)
veh_all_cols = fetch_schema("vehicles")
ppl_all_cols = fetch_schema("people")

if not veh_all_cols:
    st.info("No vehicle columns discovered (backend schema unavailable and MinIO fallback found none). "
            "You can still publish without enrichment, or check RAW_BUCKET path/env.")
if not ppl_all_cols:
    st.info("No people columns discovered (backend schema unavailable and MinIO fallback found none).")

# ---------- Common enrichment controls (function to reuse in both tabs) ----------
def enrichment_controls(prefix: str):
    st.markdown("##### Enrichment Columns")
    c1, c2 = st.columns(2)

    with c1:
        inc_v = st.checkbox("Include Vehicles", value=False, key=f"{prefix}_inc_v")
        sel_all_v = st.checkbox("Select all vehicle columns", value=False, key=f"{prefix}_all_v", disabled=not inc_v)
        veh_cols = st.multiselect(
            "Vehicles: columns",
            options=veh_all_cols,
            default=(veh_all_cols if sel_all_v else []),
            key=f"{prefix}_veh_cols",
            disabled=not inc_v,
        )

    with c2:
        inc_p = st.checkbox("Include People", value=False, key=f"{prefix}_inc_p")
        sel_all_p = st.checkbox("Select all people columns", value=False, key=f"{prefix}_all_p", disabled=not inc_p)
        ppl_cols = st.multiselect(
            "People: columns",
            options=ppl_all_cols,
            default=(ppl_all_cols if sel_all_p else []),
            key=f"{prefix}_ppl_cols",
            disabled=not inc_p,
        )

    return inc_v, veh_cols, inc_p, ppl_cols

# ---------- Streaming tab ----------
with tabs[0]:
    st.subheader("🔌 Streaming (since N days)")
    corr_id = _ts_corr()
    st.text_input("corr_id (auto)", value=corr_id, disabled=True)

    since_days = st.number_input("Since days", min_value=1, max_value=3650, value=30, step=1, key="stream_days")

    inc_v, veh_cols, inc_p, ppl_cols = enrichment_controls("stream")

    # Build payload preview
    stream_payload = {
        "type": "extract",
        "mode": "streaming",
        "corr_id": corr_id,
        "since_days": int(since_days),
        "primary": {                       # <<< ADD THIS
            "id": CRASHES_DATASET_ID,
            "alias": "crashes",
            "select": ",".join(PRIMARY_SELECT),
        },
        "enrich": make_enrich(inc_v, veh_cols, inc_p, ppl_cols),
    }

    with st.expander("Preview JSON payload", expanded=False):
        st.code(json.dumps(stream_payload, indent=2), language="json")

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        if st.button("Publish to RabbitMQ", type="primary", key="stream_pub"):
            ok, msg = publish_to_rabbit(stream_payload)
            if ok:
                st.success(f"Published: {msg}")
            else:
                st.error(f"Failed: {msg}")

    with colB:
        if st.button("Reset form", key="stream_reset"):
            reset_form([
                "stream_days",
                "stream_inc_v", "stream_all_v", "stream_veh_cols",
                "stream_inc_p", "stream_all_p", "stream_ppl_cols",
            ])
            st.rerun()

# ---------- Backfill tab ----------
with tabs[1]:
    st.subheader("🕰️ Backfill (date/time range)")
    corr_id_b = _ts_corr()
    st.text_input("corr_id (auto)", value=corr_id_b, disabled=True, key="bf_corr")

    # Dates & times
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=date.today(), key="bf_start_date")
        start_time = st.time_input("Start time", value=dtime(0,0), key="bf_start_time")
    with c2:
        end_date = st.date_input("End date", value=date.today(), key="bf_end_date")
        end_time = st.time_input("End time", value=dtime(23,59), key="bf_end_time")

    inc_v_b, veh_cols_b, inc_p_b, ppl_cols_b = enrichment_controls("backfill")

    # Compose ISO-like window strings your extractor expects (adjust if needed)
    start_ts = f"{start_date}T{start_time.strftime('%H:%M:%S')}"
    end_ts   = f"{end_date}T{end_time.strftime('%H:%M:%S')}"

    backfill_payload = {
        "type": "extract",
        "mode": "backfill",
        "corr_id": corr_id_b,
        "start": start_ts,
        "end": end_ts,
        "primary": {                       # <<< ADD THIS
            "id": CRASHES_DATASET_ID,
            "alias": "crashes",
            "select": ",".join(PRIMARY_SELECT),
        },
        "enrich": make_enrich(inc_v_b, veh_cols_b, inc_p_b, ppl_cols_b),
    }

    with st.expander("Preview JSON payload", expanded=False):
        st.code(json.dumps(backfill_payload, indent=2), language="json")

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        if st.button("Publish to RabbitMQ", type="primary", key="bf_pub"):
            ok, msg = publish_to_rabbit(backfill_payload)
            if ok:
                st.success(f"Published: {msg}")
            else:
                st.error(f"Failed: {msg}")

    with colB:
        if st.button("Reset form", key="bf_reset"):
            reset_form([
                "bf_start_date", "bf_start_time",
                "bf_end_date", "bf_end_time",
                "backfill_inc_v", "backfill_all_v", "backfill_veh_cols",
                "backfill_inc_p", "backfill_all_p", "backfill_ppl_cols",
            ])
            st.rerun()
