# cleaner/minio_io.py
import os
import io
import logging
from typing import Optional

import pandas as pd
from minio import Minio
from minio.error import S3Error

def env_to_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1","true","yes","y","on"): return True
    if v in ("0","false","no","n","off"): return False
    return default

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS   = os.getenv("MINIO_USER")
MINIO_SECRET   = os.getenv("MINIO_PASS")
MINIO_SECURE   = env_to_bool("MINIO_SSL", default=False)

class MissingObject(FileNotFoundError):
    pass

class MinioCSVSource:
    """
    Helper to fetch CSVs from MinIO as pandas DataFrames.
    """

    def __init__(self):
        self.cli = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS,
            secret_key=MINIO_SECRET,
            secure=MINIO_SECURE,
        )

    def resolve_key(self, bucket: str, prefix: str, corr_id: str) -> str:
        """
        Try both common layouts:
          1) merged/corr=<corr_id>/merged.csv    (requested in your spec)
          2) <prefix>/corr=<corr_id>/merged.csv  (how your transformer currently writes)
        Returns the first that exists, else raises MissingObject.
        """
        candidates = [
            f"merged/corr={corr_id}/merged.csv",
            f"{prefix}/corr={corr_id}/merged.csv",
        ]
        for key in candidates:
            try:
                self.cli.stat_object(bucket, key)
                return key
            except S3Error as e:
                if e.code not in {"NoSuchKey", "NotFound"}:
                    logging.debug(f"stat_object error on {key}: {e}")
                continue
        raise MissingObject(f"No merged.csv for corr={corr_id} under {candidates} in bucket={bucket}")

    def read_csv(self, bucket: str, key: str, **read_csv_kwargs) -> pd.DataFrame:
        """
        Download object into memory and parse with pandas.read_csv.
        """
        resp = None
        try:
            resp = self.cli.get_object(bucket, key)
            data = resp.read()
        finally:
            try:
                if resp:
                    resp.close()
                    resp.release_conn()
            except Exception:
                pass
        bio = io.BytesIO(data)
        return pd.read_csv(bio, **read_csv_kwargs)
