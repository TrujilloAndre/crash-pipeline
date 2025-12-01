# pages/1_Data_Management.py
import os
import io
import json
import time
import shutil
import duckdb
import pandas as pd
import streamlit as st
from typing import List, Tuple
from minio import Minio
from minio.error import S3Error

# -------------------------------
# Config (from env with defaults)
# -------------------------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_USER", "minioadmin")
MINIO_PASS     = os.getenv("MINIO_PASS", "minioadmin")
MINIO_SSL      = os.getenv("MINIO_SSL", "false").strip().lower() in ("1","true","yes","y","on")

GOLD_DB_PATH   = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")
GOLD_TABLE     = os.getenv("GOLD_TABLE", 'gold."gold"."crashes"')

BUCKET_CHOICES = ["raw-data", "transform-data"]  # adjust if you have more

# -------------------------------
# Helpers
# -------------------------------
@st.cache_resource
def minio_client() -> Minio:
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=MINIO_SSL,
    )

@st.cache_data(ttl=20)
def list_prefix(_cli, bucket: str, prefix: str):
    """Return (count, up to 200 keys) or (0, []) if any error."""
    try:
        keys, n = [], 0
        for obj in _cli.list_objects(bucket, prefix=prefix, recursive=True):
            if getattr(obj, "is_dir", False):
                continue
            n += 1
            if len(keys) < 200:
                keys.append(obj.object_name)
        return n, keys
    except Exception as e:
        # log to UI but do not raise from cached fn
        st.warning(f"MinIO preview failed: {e}")
        return 0, []



def delete_prefix(cli: Minio, bucket: str, prefix: str) -> Tuple[int, int]:
    """Delete all objects under prefix; returns (attempted, errors)."""
    attempted = 0
    errors = 0
    # MinIO has remove_objects, but we'll stream to be explicit
    for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
        if getattr(obj, "is_dir", False):
            continue
        attempted += 1
        try:
            cli.remove_object(bucket, obj.object_name)
        except S3Error:
            errors += 1
    return attempted, errors

def empty_and_delete_bucket(cli: Minio, bucket: str) -> Tuple[int, int, str]:
    """Empty bucket then delete it. Returns (removed, errors, result_msg)."""
    removed = 0
    errors = 0
    try:
        for obj in cli.list_objects(bucket, recursive=True):
            if getattr(obj, "is_dir", False):
                continue
            try:
                cli.remove_object(bucket, obj.object_name)
                removed += 1
            except S3Error:
                errors += 1
        cli.remove_bucket(bucket)
        return removed, errors, "Bucket deleted"
    except S3Error as e:
        return removed, errors, f"Failed: {e.code}"

@st.cache_data(ttl=20)
def gold_status(db_path: str) -> dict:
    info = {"exists": os.path.exists(db_path), "tables": [], "rows_total": 0}
    if not info["exists"]:
        return info
    con = duckdb.connect(db_path, read_only=True)
    # List schemas/tables (DuckDB catalogs can be named; qualify)
    df_tables = con.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type='BASE TABLE'
        ORDER BY table_schema, table_name
    """).fetchdf()
    info["tables"] = df_tables.to_dict(orient="records")
    rows_total = 0
    for row in info["tables"]:
        schema = row["table_schema"]
        table  = row["table_name"]
        try:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()[0]
        except Exception:
            cnt = None
        row["row_count"] = cnt
        rows_total += (cnt or 0)
    info["rows_total"] = rows_total
    return info

def wipe_gold_file(db_path: str) -> str:
    """Remove DB file and recreate empty DB."""
    # Ensure parent exists
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    # Recreate an empty DB
    duckdb.connect(db_path).close()
    return "Gold DB wiped and recreated."

@st.cache_data(ttl=20)
def gold_columns(db_path: str, table_fq: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    # Resolve schema and table from fully-qualified string
    # Supports forms like gold."gold"."crashes" or schema.table
    if table_fq.count('"') >= 4:
        # already quoted parts, use directly
        quoted = table_fq
    else:
        parts = [p.strip('"') for p in table_fq.split(".")]
        if len(parts) == 2:
            quoted = f'"{parts[0]}"."{parts[1]}"'
        elif len(parts) == 3:
            quoted = f'"{parts[0]}"."{parts[1]}"."{parts[2]}"'
        else:
            quoted = table_fq  # last resort

    return con.execute(f"PRAGMA table_info({quoted})").fetchdf()

def gold_preview(db_path: str, table_fq: str, columns: List[str], limit: int) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    if not columns:
        q = f"SELECT * FROM {table_fq} LIMIT {limit}"
    else:
        cols = ", ".join([f'"{c}"' for c in columns])
        q = f"SELECT {cols} FROM {table_fq} LIMIT {limit}"
    return con.execute(q).fetchdf()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Data Management", page_icon="🧰", layout="wide")
st.title("🧰 Data Management")
st.caption("Centralized admin for MinIO buckets and the Gold (DuckDB) warehouse.")

cli = minio_client()

# A) ---------------- MinIO Browser & Delete ----------------
st.subheader("A) MinIO Browser & Delete")

tab1, tab2 = st.tabs(["Delete by Folder (Prefix)", "Delete by Bucket"])

with tab1:
    st.write("Remove all objects under a specific prefix (dry-run preview required).")
    colA, colB, colC = st.columns([1, 2, 1])
    with colA:
        bucket = st.selectbox("Bucket", BUCKET_CHOICES, index=1)
    with colB:
        prefix = st.text_input("Prefix (folder)", placeholder="e.g., crash/corr=2025-10-16-23-53-19/")
    with colC:
        confirm = st.checkbox("I confirm the scope", value=False)

    # Preview (dry-run)
    if st.button("Preview (Dry-run)"):
        if not prefix:
            st.warning("Enter a prefix to preview.")
        else:
            with st.spinner("Scanning…"):
                try:
                    count, sample = list_prefix(cli, bucket, prefix)
                    st.session_state["minio_preview"] = {"bucket": bucket, "prefix": prefix, "count": count, "sample": sample}
                except S3Error as e:
                    st.error(f"MinIO error: {e}")

    # Show preview if available
    if "minio_preview" in st.session_state:
        prev = st.session_state["minio_preview"]
        st.info(f"Preview for **s3://{prev['bucket']}/{prev['prefix']}** → {prev['count']} object(s). Showing up to 200 keys:")
        if prev["sample"]:
            st.code("\n".join(prev["sample"][:200]), language="text")
        else:
            st.write("(no objects under that prefix)")

    # Delete action
    disabled = ("minio_preview" not in st.session_state) or (not confirm)
    if st.button("Delete Folder", type="primary", disabled=disabled):
        prev = st.session_state.get("minio_preview", {})
        if not prev:
            st.warning("Run Preview first.")
        else:
            with st.spinner("Deleting…"):
                attempted, errors = delete_prefix(cli, prev["bucket"], prev["prefix"])
                st.success(f"Deleted attempt(s): {attempted}, errors: {errors}")
            # clear preview so user must rescan
            st.session_state.pop("minio_preview", None)

with tab2:
    st.write("Delete an entire bucket (empties contents, then removes bucket).")
    bkt = st.selectbox("Bucket to delete", BUCKET_CHOICES, key="del_bucket")
    confirm_b = st.checkbox("I confirm this will remove ALL data in the bucket.", value=False, key="confirm_bucket")
    if st.button("Delete Bucket", type="primary", disabled=not confirm_b):
        with st.spinner("Deleting bucket…"):
            removed, errs, msg = empty_and_delete_bucket(cli, bkt)
            if "deleted" in msg.lower():
                st.success(f"{msg} | removed objects: {removed}, errors: {errs}")
            else:
                st.error(f"{msg} | removed objects: {removed}, errors: {errs}")

st.divider()

# B) ---------------- Gold Admin (DuckDB) ----------------
st.subheader("B) Gold Admin (DuckDB)")
st.caption(f"DB Path: `{GOLD_DB_PATH}`  •  Table: `{GOLD_TABLE}`")

status = gold_status(GOLD_DB_PATH)
cols = st.columns(3)
with cols[0]:
    st.metric("Gold DB Exists", "Yes" if status["exists"] else "No")
with cols[1]:
    st.metric("Total Rows (all tables)", status["rows_total"])
with cols[2]:
    st.metric("Tables", len(status["tables"]))

if status["tables"]:
    st.dataframe(pd.DataFrame(status["tables"]))

wipe_ok = st.checkbox("I confirm wiping will DELETE the on-disk DB file.", value=False)
if st.button("Wipe Gold DB (ENTIRE FILE)", type="primary", disabled=not wipe_ok):
    try:
        msg = wipe_gold_file(GOLD_DB_PATH)
        st.success(msg)
        gold_status.clear()
    except Exception as e:
        st.error(f"Failed to wipe: {e}")

st.divider()

# C) ---------------- Quick Peek (Gold — sanity view) ----------------
st.subheader("C) Quick Peek (Gold — sanity view)")
if not status["exists"]:
    st.info("Gold DB not found. Mount it into the container and/or run your cleaner first.")
else:
    try:
        schema_df = gold_columns(GOLD_DB_PATH, GOLD_TABLE)
        st.caption("Choose a few columns to preview. If left empty, the first 8 columns are auto-selected.")
        all_cols = list(schema_df["name"]) if "name" in schema_df.columns else list(schema_df["column_name"])
        default_cols = all_cols[:8]
        sel_cols = st.multiselect("Columns", options=all_cols, default=default_cols)
        limit = st.slider("Rows (limit)", min_value=0, max_value=50, value=1, step=10)

        if st.button("Preview"):
            with st.spinner("Querying DuckDB…"):
                df = gold_preview(GOLD_DB_PATH, GOLD_TABLE, sel_cols, limit)
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not read table metadata: {e}")
