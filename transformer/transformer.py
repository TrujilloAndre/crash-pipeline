# transformer/transformer.py
import os
import io
import json
import gzip
import socket
import logging
import time
import random
import traceback
from typing import List, Dict, Any

import pika
from minio import Minio
from minio.error import S3Error
import polars as pl
import re

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(level=logging.INFO, format="[transformer] %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)

# ---------------------------------
# Env / Config
# ---------------------------------
RABBIT_URL       = os.getenv("RABBITMQ_URL")
TRANSFORM_QUEUE  = os.getenv("TRANSFORM_QUEUE", "transform")
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS     = os.getenv("MINIO_USER")
MINIO_SECRET     = os.getenv("MINIO_PASS")


def env_to_bool(env_var_name: str, default: bool = False) -> bool:
    val = os.getenv(env_var_name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("true", "1", "yes", "y", "on"):
        return True
    if val in ("false", "0", "no", "n", "off"):
        return False
    return default


MINIO_SECURE = env_to_bool("MINIO_SSL", default=False)

RAW_BUCKET       = os.getenv("RAW_BUCKET")
XFORM_BUCKET_ENV = os.getenv("XFORM_BUCKET")
PREFIX           = "crash"

# ---------------------------------
# Prometheus metrics
# ---------------------------------

# capture process start once
TRANSFORMER_START_TIME = time.time()

transformer_files_written_total = Counter(
    "transformer_files_written_total",
    "Number of MinIO objects written by transformer (Silver / transform-data).",
)

transformer_app_start_time = Gauge(
    "transformer_app_start_time",
    "Unix timestamp when transformer service started",
)
transformer_app_start_time.set(TRANSFORMER_START_TIME)

# ✅ New: uptime gauge (seconds since process start)
transformer_uptime_seconds = Gauge(
    "transformer_uptime_seconds",
    "Uptime of the transformer service in seconds.",
)
# expose uptime as a function so it always reflects current uptime
transformer_uptime_seconds.set_function(lambda: time.time() - TRANSFORMER_START_TIME)

transformer_jobs_total = Counter(
    "transformer_jobs_total",
    "Total transformer jobs processed",
    ["status"],   # success, error, ignored
)

transformer_last_success_timestamp = Gauge(
    "transformer_last_success_timestamp",
    "Unix timestamp of last successful transformer job"
)

transformer_latency_seconds = Histogram(
    "transformer_latency_seconds",
    "Duration of transformer jobs in seconds",
    buckets=[0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

transformer_rows_in_total = Counter(
    "transformer_rows_in_total",
    "Total rows read by transformer from raw datasets",
)

transformer_rows_out_total = Counter(
    "transformer_rows_out_total",
    "Total rows written by transformer merged output",
)

transformer_errors_total = Counter(
    "transformer_errors_total",
    "Total errors raised while processing transform jobs",
)

# Optional aliases if you want explicit success/fail metrics
transformer_success_total = Counter(
    "transformer_success_total",
    "Number of successful transformer jobs",
)
transformer_fail_total = Counter(
    "transformer_fail_total",
    "Number of failed transformer jobs",
)

# ---------------------------------
# MinIO client
# ---------------------------------
def minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )

# ---------------------------------
# Object helpers
# ---------------------------------
def list_objects_recursive(cli: Minio, bucket: str, prefix: str) -> List[str]:
    out = []
    for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
        if getattr(obj, "is_dir", False):
            continue
        out.append(obj.object_name)
    return out


def read_json_gz_array(cli: Minio, bucket: str, key: str) -> List[Dict[str, Any]]:
    """
    Download an object and return it as a JSON array.
    Handles both gzipped (.json.gz) and plain JSON content.
    """
    resp = None
    data = b""
    try:
        resp = cli.get_object(bucket, key)
        data = resp.read()
    finally:
        try:
            if resp is not None:
                resp.close()
                resp.release_conn()
        except Exception:
            pass

    # GZIP magic header: 1F 8B
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(data)
        except OSError:
            payload = data
    else:
        payload = data

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")

    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(arr, list):
        return arr
    if isinstance(arr, dict) and isinstance(arr.get("data"), list):
        return arr["data"]
    return []


def write_csv(cli: Minio, bucket: str, key: str, df: pl.DataFrame) -> None:
    """
    Write a DataFrame as CSV to MinIO and increment the Silver object counter.
    """
    buf = io.BytesIO()
    df.write_csv(buf)
    data = buf.getvalue()

    cli.put_object(
        bucket,
        key,
        data=io.BytesIO(data),
        length=len(data),
        content_type="text/csv; charset=utf-8",
    )

    # ✅ Increment Silver object count on successful write
    transformer_files_written_total.inc()

# ---------------------------------
# Load & merge
# ---------------------------------
def _keys_for_corr(cli: Minio, bucket: str, prefix: str, dataset_alias: str, corr: str) -> List[str]:
    """Extractor writes year partitions; filter corr across them."""
    base = f"{prefix}/{dataset_alias}/"
    keys = list_objects_recursive(cli, bucket, base)
    needle = f"/corr={corr}/"
    return [k for k in keys if (k.endswith(".json.gz") or k.endswith(".json")) and needle in k]


def load_dataset(cli: Minio, raw_bucket: str, prefix: str, dataset_alias: str, corr: str) -> pl.DataFrame:
    keys = _keys_for_corr(cli, raw_bucket, prefix, dataset_alias, corr)
    rows_all: List[Dict[str, Any]] = []
    for k in keys:
        rows = read_json_gz_array(cli, raw_bucket, k)
        if rows:
            rows_all.extend(rows)
    return pl.DataFrame(rows_all) if rows_all else pl.DataFrame()


def normalize_headers_all(df: pl.DataFrame) -> pl.DataFrame:
    """Lower/strip headers and collapse internal spaces to single underscores."""
    if df.is_empty():
        return df

    def _h(c: str) -> str:
        c = c.strip().lower()
        c = re.sub(r"\s+", "_", c)
        return c

    return df.rename({c: _h(c) for c in df.columns})


def clean_string_columns(
    df: pl.DataFrame,
    *,
    lower: bool = True,
    strip: bool = True,
    collapse_spaces: bool = True,
    none_if_empty: bool = True,
) -> pl.DataFrame:
    if df.is_empty():
        return df

    exprs = []
    for name, dtype in df.schema.items():
        if dtype == pl.Utf8:
            col = pl.col(name)
            if strip:
                col = col.str.strip_chars()
            if collapse_spaces:
                col = col.str.replace_all(r"\s+", " ").str.strip_chars()
            if lower:
                col = col.str.to_lowercase()
            if none_if_empty:
                col = pl.when(col.str.len_chars() == 0).then(pl.lit(None)).otherwise(col)
            exprs.append(col.alias(name))
    return df.with_columns(exprs)


def basic_standardize(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = normalize_headers_all(df)
    df = clean_string_columns(df)
    return df.unique(maintain_order=True)


def aggregate_many_to_one(df: pl.DataFrame, id_col: str, prefix: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    keep_fields = [c for c in df.columns if c != id_col]
    text_cols = [c for c in keep_fields if df.schema.get(c, pl.Utf8) == pl.Utf8][:20]

    aggs = [pl.len().alias(f"{prefix}_count")]
    for c in text_cols:
        aggs.append(
            pl.col(c)
            .drop_nulls()
            .cast(pl.Utf8)
            .unique()
            .sort()
            .implode()
            .alias(f"{prefix}_{c}_list")
        )
    return df.group_by(id_col, maintain_order=True).agg(aggs)


def merge_crash_vehicles_people(
    crashes: pl.DataFrame,
    vehicles: pl.DataFrame,
    people: pl.DataFrame,
    id_col: str,
) -> pl.DataFrame:
    crashes = basic_standardize(crashes)
    vehicles = basic_standardize(vehicles)
    people   = basic_standardize(people)

    id_lower = id_col.lower()

    def _ensure_id(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or id_lower in df.columns:
            return df
        for c in df.columns:
            if c.lower() == id_lower:
                return df.rename({c: id_lower})
        return df

    crashes  = _ensure_id(crashes)
    vehicles = _ensure_id(vehicles)
    people   = _ensure_id(people)

    if not crashes.is_empty() and id_lower not in crashes.columns:
        return crashes

    veh_agg = aggregate_many_to_one(vehicles, id_lower, prefix="veh") if (not vehicles.is_empty() and id_lower in vehicles.columns) else pl.DataFrame()
    ppl_agg = aggregate_many_to_one(people,   id_lower, prefix="ppl") if (not people.is_empty() and id_lower in people.columns) else pl.DataFrame()

    out = crashes
    if not veh_agg.is_empty():
        out = out.join(veh_agg, on=id_lower, how="left")
    if not ppl_agg.is_empty():
        out = out.join(ppl_agg, on=id_lower, how="left")

    return out.unique(subset=[id_lower], keep="first", maintain_order=True)

# ---------------------------------
# CSV safety (for nested/array/struct cols)
# ---------------------------------
def make_csv_safe(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    def _jsonable(x):
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8")
            except Exception:
                return x.hex()
        if isinstance(x, (list, tuple, set)):
            return [_jsonable(v) for v in list(x)]
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        if hasattr(x, "to_list"):
            try:
                return [_jsonable(v) for v in x.to_list()]
            except Exception:
                pass
        if hasattr(x, "to_dict"):
            try:
                return {k: _jsonable(v) for k, v in x.to_dict().items()}
            except Exception:
                pass
        return str(x)

    fixes, drop_cols = [], []
    for name, dtype in df.schema.items():
        if isinstance(dtype, (pl.List, pl.Struct)) or dtype.__class__.__name__ == "Array":
            fixes.append(
                pl.col(name)
                  .map_elements(
                      lambda x: json.dumps(_jsonable(x), ensure_ascii=False),
                      return_dtype=pl.String,
                  )
                  .alias(f"{name}_json")
            )
            drop_cols.append(name)

    if not fixes:
        return df
    out = df.with_columns(fixes)
    return out.drop(drop_cols) if drop_cols else out

# ---------------------------------
# Transform runner (writes CSV)
# ---------------------------------
def run_transform_job(msg: dict):
    corr       = msg.get("corr_id")
    raw_bucket = msg.get("raw_bucket", RAW_BUCKET)
    out_bucket = msg.get("xform_bucket") or msg.get("clean_bucket") or XFORM_BUCKET_ENV
    prefix = PREFIX
    if not corr or not out_bucket:
        raise ValueError("run_transform_job: missing corr_id or (xform_bucket|clean_bucket|XFORM_BUCKET)")

    cli = minio_client()

    # Ensure target bucket exists
    try:
        if not cli.bucket_exists(out_bucket):
            cli.make_bucket(out_bucket)
    except S3Error as e:
        if e.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            raise

    # Load raw pages
    crashes_df  = load_dataset(cli, raw_bucket, prefix, "crashes",  corr)
    vehicles_df = load_dataset(cli, raw_bucket, prefix, "vehicles", corr)
    people_df   = load_dataset(cli, raw_bucket, prefix, "people",   corr)

    # Metrics: rows in
    rows_in = (
        (0 if crashes_df is None else crashes_df.height)
        + (0 if vehicles_df is None else vehicles_df.height)
        + (0 if people_df is None else people_df.height)
    )
    transformer_rows_in_total.inc(rows_in)

    merged = merge_crash_vehicles_people(
        crashes=crashes_df,
        vehicles=vehicles_df,
        people=people_df,
        id_col="crash_record_id",
    )

    # Metrics: rows out
    transformer_rows_out_total.inc(merged.height)

    out_key = f"{prefix}/corr={corr}/merged.csv"
    write_csv(cli, out_bucket, out_key, make_csv_safe(merged))
    logging.info(f"Wrote s3://{out_bucket}/{out_key} (rows={merged.height}, cols={merged.width})")

    # ✅ update last successful job timestamp
    transformer_last_success_timestamp.set(time.time())

# ---------------------------------
# RabbitMQ consumer
# ---------------------------------
def wait_for_port(host: str, port: int, tries: int = 60, delay: float = 1.0):
    for _ in range(tries):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(delay)
    return False


def start_consumer():
    from pika.exceptions import AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError

    params = pika.URLParameters(RABBIT_URL)

    host = params.host or "rabbitmq"
    port = params.port or 5672
    if not wait_for_port(host, port, tries=60, delay=1.0):
        raise SystemExit(f"[transformer] RabbitMQ not reachable at {host}:{port} after waiting.")

    max_tries = 60
    base_delay = 1.5
    conn = None

    for i in range(1, max_tries + 1):
        try:
            conn = pika.BlockingConnection(params)
            break
        except (AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError) as e:
            if i == 1:
                logging.info(f"Waiting for RabbitMQ @ {RABBIT_URL} …")
            if i % 10 == 0:
                logging.info(f"Still waiting (attempt {i}/{max_tries}): {e.__class__.__name__}")
            time.sleep(base_delay + random.random())

    if conn is None or not conn.is_open:
        raise SystemExit("[transformer] Could not connect to RabbitMQ after multiple attempts.")

    ch = conn.channel()
    ch.queue_declare(queue=TRANSFORM_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_msg(chx, method, props, body):
        job_start = time.time()
        status = "success"
        try:
            msg = json.loads(body.decode("utf-8"))
            mtype = msg.get("type", "")
            if mtype not in ("transform", "clean"):
                logging.info(f"ignoring message type={mtype!r}")
                transformer_jobs_total.labels(status="ignored").inc()
                chx.basic_ack(delivery_tag=method.delivery_tag)
                return

            logging.info(f"Received transform job (type={mtype}) corr={msg.get('corr_id')}")
            try:
                run_transform_job(msg)
            except Exception:
                status = "error"
                transformer_errors_total.inc()
                raise
            finally:
                dt = time.time() - job_start
                transformer_latency_seconds.observe(dt)
                transformer_jobs_total.labels(status=status).inc()
                if status == "success":
                    transformer_success_total.inc()
                else:
                    transformer_fail_total.inc()

            chx.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            traceback.print_exc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    logging.info(f"Up. Waiting for jobs on queue '{TRANSFORM_QUEUE}'")
    ch.basic_consume(queue=TRANSFORM_QUEUE, on_message_callback=on_msg)
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        try:
            ch.stop_consuming()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def main():
    # Start Prometheus HTTP server on :8000
    start_http_server(8000)
    logging.info("Prometheus metrics server listening on :8000/metrics")
    start_consumer()

if __name__ == "__main__":
    main()
