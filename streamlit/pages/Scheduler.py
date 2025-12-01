# pages/3_Scheduler.py
# Streamlit Scheduler: create/list/delete cron schedules for pipeline runs
import os
import json
from datetime import time as dtime

import requests
import streamlit as st

# ---------------------------------
# Config (env)
# ---------------------------------
API_BASE = os.getenv("SCHEDULER_BASE", os.getenv("API_BASE", "http://backend:8080"))
SCHEDULES_PATH  = os.getenv("SCHEDULES_PATH", "/api/schedules")  # GET/POST list & create
SCHEDULE_ID_FMT = os.getenv("SCHEDULE_ID_FMT", "/api/schedules/{id}")  # DELETE
DEFAULT_SINCE_DAYS = int(os.getenv("DEFAULT_SINCE_DAYS", "30"))  # streaming window
TZ              = os.getenv("SCHED_TIMEZONE", "America/Chicago")  # informational

# If your scheduler service is behind auth, set API_AUTH_USER/PASS
API_AUTH_USER = os.getenv("API_AUTH_USER")
API_AUTH_PASS = os.getenv("API_AUTH_PASS")
AUTH = (API_AUTH_USER, API_AUTH_PASS) if API_AUTH_USER and API_AUTH_PASS else None

# ---------------------------------
# Helpers
# ---------------------------------
def build_cron(freq: str, at_time: dtime, weekday: str | None) -> str:
    """
    Returns standard 5-field cron: 'm H dom mon dow'
    Daily @ HH:MM  -> 'MM HH * * *'
    Weekly day @ T -> 'MM HH * * DOW'  where DOW = 0-6 (Sun=0)
    Custom         -> return as-is (already validated upstream)
    """
    mm = at_time.minute
    hh = at_time.hour
    if freq == "Daily":
        return f"{mm} {hh} * * *"
    if freq == "Weekly":
        # Map human weekday to cron number
        dow_map = {
            "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
            "Thursday": 4, "Friday": 5, "Saturday": 6
        }
        dow = dow_map.get(weekday or "Monday", 1)
        return f"{mm} {hh} * * {dow}"
    # Should not reach for Custom (handled elsewhere)
    return f"{mm} {hh} * * *"

def validate_cron(expr: str) -> bool:
    """Very light sanity check: 5 fields, not empty."""
    parts = expr.split()
    return len(parts) == 5 and all(len(p) > 0 for p in parts)

def create_schedule(payload: dict) -> tuple[bool, str]:
    try:
        url = f"{API_BASE}{SCHEDULES_PATH}"
        r = requests.post(url, json=payload, auth=AUTH, timeout=10)
        if r.ok:
            return True, "Created"
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=5)
def list_schedules() -> tuple[bool, list, str]:
    try:
        url = f"{API_BASE}{SCHEDULES_PATH}"
        r = requests.get(url, auth=AUTH, timeout=10)
        if not r.ok:
            return False, [], f"HTTP {r.status_code}: {r.text}"
        data = r.json()
        # Accept both list or {"items":[...]}
        items = data if isinstance(data, list) else data.get("items", [])
        return True, items, ""
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"

def delete_schedule(sched_id: str) -> tuple[bool, str]:
    try:
        url = f"{API_BASE}{SCHEDULE_ID_FMT.format(id=sched_id)}"
        r = requests.delete(url, auth=AUTH, timeout=10)
        if r.ok:
            return True, "Deleted"
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ---------------------------------
# UI
# ---------------------------------
st.set_page_config(page_title="Scheduler", page_icon="⏰", layout="wide")
st.title("⏰ Scheduler")
st.caption("Automate streaming runs with cron schedules. Times interpreted in "
           f"**{TZ}** (server/worker timezone).")

with st.container(border=True):
    st.subheader("Create Schedule")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        freq = st.radio("Frequency", ["Daily", "Weekly", "Custom cron"], horizontal=False)
    with col2:
        at_time = st.time_input("Run at (HH:MM)", value=dtime(9, 0))
        weekday = st.selectbox("Day of week (for Weekly)", 
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                               index=0, disabled=(freq != "Weekly"))
    with col3:
        custom_cron = st.text_input("Custom cron (m H dom mon dow)", placeholder="e.g. 0 6 * * 1-5", disabled=(freq != "Custom cron"))

    # Config Type (fixed to streaming per your spec)
    st.markdown("**Config Type:** `streaming`")
    since_days = st.number_input("Since days (streaming window)", min_value=1, max_value=3650,
                                 value=DEFAULT_SINCE_DAYS, step=1)

    # Compose cron
    if freq == "Custom cron":
        cron_expr = custom_cron.strip()
    else:
        cron_expr = build_cron(freq, at_time, weekday if freq == "Weekly" else None)

    # Preview payload (typical minimal config; extend as needed)
    preview = {
        "cron": cron_expr,
        "timezone": TZ,
        "config": {
            "type": "extract",
            "mode": "streaming",
            "since_days": int(since_days),
            # You can add your primary/enrich defaults here if your scheduler triggers extractor directly:
            # "primary": { "id": os.getenv("CRASHES_DATASET_ID","85ca-t3if"), "alias":"crashes",
            #             "select": "crash_record_id,crash_date,weather_condition,lighting_condition,posted_speed_limit" },
            # "enrich": [...]
        },
        "enabled": True
    }

    with st.expander("Preview schedule payload", expanded=False):
        st.code(json.dumps(preview, indent=2), language="json")

    valid = validate_cron(cron_expr)
    if not valid:
        st.warning("Cron expression looks invalid (needs 5 space-separated fields).")

    if st.button("Create Schedule", type="primary", disabled=not valid):
        ok, msg = create_schedule(preview)
        if ok:
            st.success(msg)
            list_schedules.clear()
            st.rerun()
        else:
            st.error(f"Failed to create: {msg}")

st.divider()

st.subheader("Active Schedules")
ok, items, err = list_schedules()
if not ok:
    st.error(f"Could not load schedules: {err}")
else:
    if not items:
        st.info("No active schedules.")
    else:
        # Render table with delete controls
        # Normalize a few common fields
        def norm(row, key, default=""):
            return row.get(key, default)

        headers = st.columns([2, 2, 2, 2, 1])
        headers[0].markdown("**ID**")
        headers[1].markdown("**Cron**")
        headers[2].markdown("**Config (short)**")
        headers[3].markdown("**Last Run**")
        headers[4].markdown("**Actions**")

        for row in items:
            sched_id  = str(norm(row, "id") or norm(row, "schedule_id") or norm(row, "_id") or "")
            cron      = norm(row, "cron")
            last_run  = norm(row, "last_run_at") or norm(row, "lastRunAt") or "—"
            cfg       = norm(row, "config", {})
            cfg_short = f"{cfg.get('mode', '?')} | since_days={cfg.get('since_days','?')}" if isinstance(cfg, dict) else str(cfg)

            cols = st.columns([2, 2, 2, 2, 1])
            cols[0].code(sched_id or "(no id)", language="text")
            cols[1].code(cron, language="text")
            cols[2].write(cfg_short)
            cols[3].write(last_run)

            with cols[4]:
                if st.button("🗑️", key=f"del_{sched_id}", help="Delete schedule"):
                    okd, msg = delete_schedule(sched_id)
                    if okd:
                        st.success("Deleted")
                        list_schedules.clear()
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {msg}")

st.caption("Tip: For weekly jobs at 6:30 AM every Mon–Fri, use cron `30 6 * * 1-5`. Times are interpreted by the scheduler service.")
