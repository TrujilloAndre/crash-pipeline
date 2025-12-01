# cleaner/cleaner.py
import os
import json
import logging
import time
import random
import traceback

import pika
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from minio_io import MinioCSVSource, MissingObject
from duckdb_writer import DuckDBWriter

logging.basicConfig(level=logging.INFO, format="[cleaner] %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)

# ---------------------------------
# Prometheus Metrics
# ---------------------------------
_START_TIME = time.time()

# Overall jobs (with status label so you can see success vs error)
cleaner_jobs_total = Counter(
    "cleaner_jobs_total",
    "Total number of cleaner jobs processed, by status.",
    ["status"],  # "success" | "error"
)

cleaner_gold_rows_total = Gauge(
    "cleaner_gold_rows_total",
    "Total rows in the DuckDB gold table after last successful cleaner upsert.",
)

cleaner_rows_dropped_total = Counter(
    "cleaner_rows_dropped_total",
    "Total number of rows dropped during cleaning."
)

cleaner_last_success_timestamp = Gauge(
    "cleaner_last_success_timestamp",
    "Unix timestamp of last successful cleaner job"
)

# Explicit success/fail counters (handy for dashboards)
cleaner_success_total = Counter(
    "cleaner_success_total",
    "Number of successful cleaner jobs.",
)

cleaner_fail_total = Counter(
    "cleaner_fail_total",
    "Number of failed cleaner jobs.",
)

# Job-level duration
cleaner_job_duration_seconds = Histogram(
    "cleaner_job_duration_seconds",
    "Duration of cleaner jobs in seconds.",
    buckets=[0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# Total rows processed (incoming) by cleaner
cleaner_rows_processed_total = Counter(
    "cleaner_rows_processed_total",
    "Total number of rows processed (incoming) by the cleaner.",
)

# Per-stage durations
cleaner_stage_duration_seconds = Histogram(
    "cleaner_stage_duration_seconds",
    "Duration of main cleaner stages in seconds.",
    ["stage"],  # "locate", "load", "clean", "upsert"
)

# Errors by stage
cleaner_errors_total = Counter(
    "cleaner_errors_total",
    "Total number of cleaner errors, by stage.",
    ["stage"],  # "consumer", "missing_object", "job"
)

# Uptime gauge
cleaner_uptime_seconds = Gauge(
    "cleaner_uptime_seconds",
    "Uptime of the cleaner service in seconds.",
)

cleaner_gold_db_file_bytes = Gauge(
    "cleaner_gold_db_file_bytes",
    "On-disk size in bytes of the DuckDB gold DuckDB file.",
)

# ---------------------------------
# RabbitMQ config
# ---------------------------------
RABBIT_URL  = os.getenv("RABBITMQ_URL")
CLEAN_QUEUE = os.getenv("CLEAN_QUEUE", "clean")

# Import your cleaning rules. Expect clean_df(df: pd.DataFrame) -> pd.DataFrame.
try:
    from cleaning_rules import clean_df  # you can implement this in your file
except Exception:
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        # Fallback: pass-through
        return df

# ---------------------------------
# Helpers
# ---------------------------------
def _parse_msg(body: bytes) -> dict:
    s = body.decode("utf-8", errors="replace").strip()
    if not s:
        raise ValueError("Empty message body")

    # Normal case
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Sometimes payload is a JSON-encoded JSON string: "\"{...}\""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        try:
            return json.loads(json.loads(s))
        except Exception:
            pass

    logging.error(f"Bad message body (first 200 bytes): {s[:200]!r}")
    raise ValueError("Could not parse message as JSON")

# ---------------------------------
# Consumer
# ---------------------------------
def start_consumer():
    params = pika.URLParameters(RABBIT_URL)
    conn = None

    # Backoff while RabbitMQ comes up
    for i in range(1, 61):
        try:
            conn = pika.BlockingConnection(params)
            break
        except Exception as e:
            if i == 1:
                logging.info(f"Waiting for RabbitMQ @ {RABBIT_URL} …")
            if i % 10 == 0:
                logging.info(f"Still waiting (attempt {i}/60): {e.__class__.__name__}")
            time.sleep(1.5 + random.random())

    if conn is None or not conn.is_open:
        cleaner_errors_total.labels(stage="consumer").inc()
        raise SystemExit("[cleaner] Could not connect to RabbitMQ.")

    ch = conn.channel()
    ch.queue_declare(queue=CLEAN_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_msg(chx, method, props, body):
        job_start = time.time()
        status = "success"
        rows_in = 0

        try:
            # -----------------------
            # 0) Parse message
            # -----------------------
            msg = _parse_msg(body)
            if msg.get("type") != "clean":
                logging.info(f"Ignoring message type={msg.get('type')!r}")
                chx.basic_ack(delivery_tag=method.delivery_tag)
                return

            corr_id      = msg["corr_id"]
            xform_bucket = msg["xform_bucket"]
            prefix       = msg.get("prefix", "crash")
            gold_db_path = msg["gold_db_path"]
            gold_table   = msg["gold_table"]  # e.g. "gold.crashes"

            logging.info(
                f"Clean job corr={corr_id} bucket={xform_bucket} "
                f"prefix={prefix} → {gold_table}"
            )

            # -----------------------
            # 1) Locate Silver CSV in MinIO
            # -----------------------
            t0 = time.time()
            src = MinioCSVSource()
            key = src.resolve_key(xform_bucket, prefix, corr_id)
            cleaner_stage_duration_seconds.labels(stage="locate").observe(time.time() - t0)
            logging.info(f"Found merged CSV at s3://{xform_bucket}/{key}")

            # -----------------------
            # 2) Load CSV
            # -----------------------
            t0 = time.time()
            df = src.read_csv(xform_bucket, key)
            cleaner_stage_duration_seconds.labels(stage="load").observe(time.time() - t0)
            rows_in = len(df)

            # -----------------------
            # 3) Clean (your rules)
            # -----------------------
            t0 = time.time()
            df_clean = clean_df(df)
            cleaner_stage_duration_seconds.labels(stage="clean").observe(time.time() - t0)

            dropped = max(0, len(df) - len(df_clean))
            if dropped > 0:
                cleaner_rows_dropped_total.inc(dropped)

            # -----------------------
            # 4) Upsert → DuckDB gold
            # -----------------------
            t0 = time.time()
            writer = DuckDBWriter(gold_db_path, gold_table, key_col="crash_record_id")
            result = writer.upsert(df_clean)
            cleaner_stage_duration_seconds.labels(stage="upsert").observe(time.time() - t0)

            incoming_rows = result.get("incoming_rows", rows_in or 0)
            rows_in = incoming_rows or rows_in
            total_after = result.get("total_after")
            if isinstance(total_after, int):
                cleaner_gold_rows_total.set(total_after)

            try:
                size_bytes = os.path.getsize(gold_db_path)
                cleaner_gold_db_file_bytes.set(size_bytes)
            except OSError:
                # file might not exist yet or path invalid; don't crash metrics for that
                pass
            
            logging.info(
                f"corr={corr_id} upsert summary: "
                f"received={result['incoming_rows']} inserted={result['inserted_rows']} "
                f"updated={result['updated_rows']} unchanged={result['unchanged_rows']} "
                f"total={result['total_after']}"
            )

            chx.basic_ack(delivery_tag=method.delivery_tag)

        except MissingObject as e:
            status = "error"
            cleaner_errors_total.labels(stage="missing_object").inc()
            logging.error(f"Missing merged.csv: {e}")
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except Exception:
            status = "error"
            cleaner_errors_total.labels(stage="job").inc()
            traceback.print_exc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        finally:
            # Per-job metrics
            duration = time.time() - job_start
            cleaner_job_duration_seconds.observe(duration)
            cleaner_jobs_total.labels(status=status).inc()
            cleaner_uptime_seconds.set(time.time() - _START_TIME)

            if status == "success":
                cleaner_success_total.inc()
                cleaner_last_success_timestamp.set(time.time())
            else:
                cleaner_fail_total.inc()

            if rows_in > 0:
                cleaner_rows_processed_total.inc(rows_in)

    logging.info(f"Up. Waiting for jobs on queue '{CLEAN_QUEUE}'")
    ch.basic_consume(queue=CLEAN_QUEUE, on_message_callback=on_msg)
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

# ---------------------------------
# Main
# ---------------------------------
if __name__ == "__main__":
    # Prometheus metrics HTTP endpoint
    start_http_server(8000)
    logging.info("[cleaner] Prometheus metrics available on :8000/metrics")
    start_consumer()
