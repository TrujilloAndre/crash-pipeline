# pages/Reports.py
import os
import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st
import duckdb
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    HAVE_RL = True
except Exception:
    HAVE_RL = False

# Optional MinIO (for finding latest corrid & run history)
try:
    from minio import Minio
    HAVE_MINIO = True
except Exception:
    HAVE_MINIO = False

# -----------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------
GOLD_DB   = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")
GOLD_TBL  = os.getenv("GOLD_TABLE", 'gold."gold"."crashes"')

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_USER")
MINIO_PASS     = os.getenv("MINIO_PASS")
MINIO_SSL      = os.getenv("MINIO_SSL", "false")
XFORM_BUCKET   = os.getenv("XFORM_BUCKET", "transform-data")
RUN_PREFIX     = os.getenv("RUN_PREFIX", "crash")  # where merged.csv lives

# -----------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------
def env_to_bool(s: str) -> bool:
    if s is None:
        return False
    return str(s).strip().lower() in ("1","true","yes","y","on")

def duck_qdf(sql: str, params=None) -> pd.DataFrame:
    """Fresh read-only connection per query; returns DataFrame."""
    if not os.path.exists(GOLD_DB):
        return pd.DataFrame()
    con = duckdb.connect(GOLD_DB, read_only=True)
    try:
        return con.execute(sql, params or []).fetchdf()
    finally:
        con.close()

def get_gold_summary() -> dict:
    out = {
        "row_count": None,
        "latest_crash_dt": None,
        "last_run_ts": None,  # placeholder if you later record this in a table
    }
    if not os.path.exists(GOLD_DB):
        return out

    # row count
    try:
        df = duck_qdf(f"SELECT COUNT(*) AS n FROM {GOLD_TBL}")
        if not df.empty:
            out["row_count"] = int(df.loc[0,"n"])
    except Exception:
        pass

    # detect a timestamp column if present
    try:
        info = duck_qdf(f"PRAGMA table_info({GOLD_TBL})")
        ts_col = None
        if not info.empty:
            names = [str(x) for x in info["name"].tolist()]
            for cand in ["crash_datetime","crash_date","datetime","date_time","ts"]:
                for real in names:
                    if real.lower() == cand.lower():
                        ts_col = real
                        break
                if ts_col: break
        if ts_col:
            mx = duck_qdf(f"SELECT MAX({ts_col}) AS mx FROM {GOLD_TBL}")
            if not mx.empty and pd.notnull(mx.loc[0,"mx"]):
                out["latest_crash_dt"] = str(mx.loc[0,"mx"])
    except Exception:
        pass

    return out

def minio_client():
    if not HAVE_MINIO or not MINIO_USER or not MINIO_PASS:
        return None
    try:
        return Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_USER,
            secret_key=MINIO_PASS,
            secure=env_to_bool(MINIO_SSL),
        )
    except Exception:
        return None

_CORR_RE = re.compile(r"corr=([^/]+)/merged\.csv$")

def scan_run_history():
    """
    Build a lightweight 'run history' by scanning MinIO for:
      {RUN_PREFIX}/corr=<corrid>/merged.csv
    Returns a DataFrame: corrid, object, size, last_modified
    """
    cli = minio_client()
    if cli is None:
        return pd.DataFrame(columns=["corrid","object","size","last_modified"])

    objs = []
    # Look under both "<prefix>/" and "merged/" just in case your transformer wrote either
    candidates = [f"{RUN_PREFIX}/", "merged/"]
    for base in candidates:
        try:
            for o in cli.list_objects(XFORM_BUCKET, prefix=base, recursive=True):
                name = getattr(o, "object_name", "")
                if name.endswith("merged.csv"):
                    m = _CORR_RE.search(name)
                    corrid = m.group(1) if m else None
                    objs.append({
                        "corrid": corrid,
                        "object": name,
                        "size": getattr(o, "size", None),
                        "last_modified": getattr(o, "last_modified", None),
                    })
        except Exception:
            # continue other prefixes
            pass

    df = pd.DataFrame(objs)
    if df.empty:
        return df

    # Normalize timestamps to string local-ish
    def fmt_ts(ts):
        try:
            # MinIO returns aware datetime; show ISO without microseconds
            return ts.astimezone().replace(microsecond=0).isoformat()
        except Exception:
            return str(ts)

    if "last_modified" in df.columns:
        df["last_modified"] = df["last_modified"].apply(fmt_ts)

    # Try to sort corrid lexicographically (your corr format is sortable)
    df = df.sort_values(["corrid","last_modified"], ascending=[False, False], na_position="last")
    return df

def get_latest_corrid(df_runs: pd.DataFrame) -> str | None:
    if df_runs is None or df_runs.empty:
        return None
    for c in df_runs["corrid"].tolist():
        if isinstance(c, str) and len(c) > 0:
            return c
    return None

def df_to_download(name: str, df: pd.DataFrame):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {name} (CSV)",
        data=csv,
        file_name=f"{name.replace(' ','_').lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )

def build_pdf_report(gold_summary: dict, runs_df: pd.DataFrame, latest_corr: str, latest_rows: pd.DataFrame, gold_tbl: str) -> bytes:
    """
    Create a 1–2 page PDF summary.
    """
    if not HAVE_RL:
        raise RuntimeError("ReportLab is not installed. Add 'reportlab' to requirements and rebuild.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    x = margin
    y = height - margin

    def line(txt, dy=14, bold=False):
        nonlocal y
        if bold:
            c.setFont("Helvetica-Bold", 11)
        else:
            c.setFont("Helvetica", 11)
        c.drawString(x, y, txt)
        y -= dy

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Crash ML — Pipeline Report")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - margin, y, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    y -= 24

    # Gold summary
    line("Gold Summary", bold=True)
    line(f"Gold table: {gold_tbl}")
    rc = gold_summary.get("row_count")
    ld = gold_summary.get("latest_crash_dt")
    line(f"Row count: {rc if rc is not None else '—'}")
    line(f"Latest data date in Gold: {ld if ld else '—'}")
    y -= 6

    # Latest corr
    line("Latest Run", bold=True)
    line(f"Latest corrid: {latest_corr if latest_corr else '—'}")
    # If we have at least one object row, show last_modified from first
    if latest_rows is not None and not latest_rows.empty:
        lm = latest_rows.iloc[0].get("last_modified") or latest_rows.iloc[0].get("object_last_modified")
    else:
        lm = None
    line(f"Last run timestamp: {lm if lm else '—'}")
    y -= 8

    # Latest run objects table (up to 10)
    if latest_rows is not None and not latest_rows.empty:
        # normalize column names for display
        disp = latest_rows.rename(columns={
            "object": "object_path",
            "size": "size_bytes",
            "last_modified": "last_modified"
        }).copy()

        cols = [c for c in ["object_path", "size_bytes", "last_modified"] if c in disp.columns]
        disp = disp[cols].head(10)

        line("Latest Run Artifacts:", bold=True)
        c.setFont("Helvetica", 9)
        # column headers
        y -= 2
        c.drawString(x, y, "object_path")
        c.drawString(x + 3.7*inch, y, "size")
        c.drawString(x + 5.0*inch, y, "last_modified")
        y -= 12
        # rows
        for _, r in disp.iterrows():
            obj = str(r.get("object_path", ""))[:60]
            size = str(r.get("size_bytes", ""))
            lm   = str(r.get("last_modified", ""))
            c.drawString(x, y, obj)
            c.drawRightString(x + 4.7*inch, y, size)
            c.drawString(x + 5.0*inch, y, lm)
            y -= 12
            if y < margin + 36:
                c.showPage()
                y = height - margin
    else:
        line("No artifacts found for the latest run.")

    # Footer
    y = max(y, margin + 18)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, margin, "Generated by Reports — Streamlit")
    c.showPage()
    c.save()
    return buf.getvalue()


# -----------------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------------
st.set_page_config(page_title="Reports", page_icon="📑", layout="wide")
st.title("📑 Reports")
st.caption("Summarized metrics and health of the ETL process over time.")

# ---------- Summary cards ----------
runs_df = scan_run_history()
latest_corr = get_latest_corrid(runs_df)
gold = get_gold_summary()

cards = st.columns(5)
with cards[0]:
    st.markdown("**Total runs**")
    st.metric(label="", value=f"{0 if runs_df is None or runs_df.empty else len(runs_df):,}")
with cards[1]:
    st.markdown("**Latest corrid**")
    if latest_corr:
        st.code(latest_corr)
    else:
        st.write("—")
with cards[2]:
    st.markdown("**Gold row count**")
    st.metric(label="", value=f"{gold['row_count']:,}" if gold["row_count"] is not None else "—")
with cards[3]:
    st.markdown("**Latest data date in Gold**")
    st.write(gold["latest_crash_dt"] or "—")
with cards[4]:
    st.markdown("**Last run timestamp**")
    # If you later store run end times in a table, surface here; for now use latest object time
    last_ts = None
    if runs_df is not None and not runs_df.empty:
        last_ts = runs_df.iloc[0]["last_modified"]
    st.write(last_ts or "—")

st.divider()

# ---------- Latest Run Summary ----------
st.subheader("Latest Run Summary")
if latest_corr:
    # Filter for the latest corrid merged.csv objects
    latest_rows = runs_df[runs_df["corrid"] == latest_corr].copy()
    # "Config used" is unknown without a control table; we can display 'unknown' or infer by path.
    config_used = "unknown (streaming/backfill not recorded)"
    st.markdown(f"""
**CorrID:** `{latest_corr}`  
**Config:** {config_used}  
**Artifacts**: MinIO → bucket `{XFORM_BUCKET}`, object(s) under `{RUN_PREFIX}/corr={latest_corr}/.../merged.csv`
""")

    st.dataframe(
        latest_rows.rename(columns={
            "object":"object_path",
            "size":"object_size_bytes",
            "last_modified":"object_last_modified"
        }),
        use_container_width=True, height=200
    )
else:
    st.info("No runs found in MinIO (no merged.csv objects detected).")

st.divider()

# ---------- Download Reports ----------
st.subheader("Download Reports")

colA, colB, colC = st.columns(3)

# A) Run history CSV (corrid, object, size, last_modified)
with colA:
    st.markdown("**Run history**")
    if runs_df is not None and not runs_df.empty:
        df_to_download("Run History", runs_df[["corrid","object","size","last_modified"]])
    else:
        st.write("No data.")

# B) Gold snapshot CSV (table, row count, latest date)
with colB:
    st.markdown("**Gold snapshot**")
    snap = pd.DataFrame([{
        "table": GOLD_TBL,
        "row_count": gold["row_count"],
        "latest_data_date": gold["latest_crash_dt"]
    }])
    df_to_download("Gold Snapshot", snap)

# C) Errors summary CSV (placeholder)
with colC:
    st.markdown("**Errors summary**")
    # If you start writing error events to a gold.run_events table, query & export it here.
    errs = pd.DataFrame(columns=["corrid","type","message_count"])
    df_to_download("Errors Summary", errs)

st.markdown("**Report (PDF)**")
try:
    latest_rows = runs_df[runs_df["corrid"] == latest_corr].copy() if latest_corr else pd.DataFrame()
    pdf_bytes = build_pdf_report(gold, runs_df, latest_corr, latest_rows, GOLD_TBL)
    st.download_button(
        label="⬇️ Download report.pdf",
        data=pdf_bytes,
        file_name="report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
except Exception as e:
    st.warning(f"Could not generate PDF: {e}")


st.caption("Tip: For richer reports, persist run metadata (mode, window, row counts, warnings) into a `gold.run_history` table during your Cleaner/Transformer stages.")
