# cleaner/duckdb_writer.py
import os
import logging
from typing import Dict, List

import duckdb
import pandas as pd


def qident(name: str) -> str:
    """Quote an identifier for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


class DuckDBWriter:
    """
    Idempotent upsert into a DuckDB gold table.
    - Fully qualifies as: <catalog>."<schema>"."<table>"
    - Adds missing columns based on incoming dataframe dtypes
    - Deletes then inserts within a transaction (idempotent by key)
    """

    def __init__(self, db_path: str, table: str, key_col: str = "crash_record_id"):
        self.db_path = db_path
        self.key_col = key_col

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(db_path)

        # Resolve catalog (usually 'main')
        catalog = "main"
        try:
            for _, name, _ in self.con.execute("PRAGMA database_list").fetchall():
                if name != "temp":
                    catalog = name
                    break
        except Exception:
            pass

        # Split "schema.table" or default to schema "gold"
        if "." in table:
            schema, tbl = table.split(".", 1)
        else:
            schema, tbl = "gold", table

        self.catalog = catalog
        self.schema = schema
        self.tbl = tbl

        self.fq_schema = f"{self.catalog}.{qident(self.schema)}"
        self.fq_table = f"{self.catalog}.{qident(self.schema)}.{qident(self.tbl)}"

        # Create schema & minimal table if absent
        self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {self.fq_schema};")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS "
            + self.fq_table
            + " ("
            + " crash_record_id VARCHAR PRIMARY KEY,"
            + " crash_date TIMESTAMP,"
            + " weather_condition VARCHAR,"
            + " lighting_condition VARCHAR,"
            + " veh_count INTEGER,"
            + " ppl_count INTEGER,"
            + " injuries_total INTEGER"
            + " );"
        )

    # ---------- helpers ----------
    def _infer_duck_type(self, s: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(s):
            return "BOOLEAN"
        if pd.api.types.is_integer_dtype(s):
            return "INTEGER"
        if pd.api.types.is_float_dtype(s):
            return "DOUBLE"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "TIMESTAMP"
        return "VARCHAR"

    def _existing_columns(self) -> List[str]:
        sql = (
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_catalog = ? AND table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position"
        )
        return [r[0] for r in self.con.execute(sql, [self.catalog, self.schema, self.tbl]).fetchall()]

    def _ensure_columns(self, df: pd.DataFrame):
        existing = set(self._existing_columns())
        for col in df.columns:
            if col not in existing:
                duck_type = self._infer_duck_type(df[col])
                self.con.execute(
                    "ALTER TABLE " + self.fq_table + " ADD COLUMN " + qident(col) + " " + duck_type + ";"
                )

    # ---------- upsert ----------
    def upsert(self, df_in: pd.DataFrame) -> Dict[str, int]:
        if df_in.empty:
            total = self.con.execute("SELECT COUNT(*) FROM " + self.fq_table).fetchone()[0]
            return {
                "incoming_rows": 0,
                "inserted_rows": 0,
                "updated_rows": 0,
                "unchanged_rows": 0,
                "total_after": int(total),
            }

        # normalize column names
        df = df_in.copy()
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        )

        if self.key_col not in df.columns:
            raise ValueError(f"Incoming frame missing key column '{self.key_col}'")

        df = df[df[self.key_col].notna()]

        # Ensure columns exist
        self._ensure_columns(df)

        # Stage incoming
        self.con.register("clean_incoming", df)

        incoming_rows = int(self.con.execute("SELECT COUNT(*) FROM clean_incoming").fetchone()[0])
        existing_rows = int(
            self.con.execute(
                "SELECT COUNT(*) FROM "
                + self.fq_table
                + " t JOIN clean_incoming i USING ("
                + self.key_col
                + ")"
            ).fetchone()[0]
        )

        cols = self._existing_columns()

        # Build INSERT column list and SELECT projection
        insert_cols = ", ".join(qident(c) for c in cols)
        select_expr_parts = []
        for c in cols:
            if c in df.columns:
                select_expr_parts.append("i." + qident(c))
            else:
                select_expr_parts.append("NULL")
        select_expr = ", ".join(select_expr_parts)

        # Transaction: delete then insert
        self.con.execute("BEGIN")
        try:
            self.con.execute(
                "DELETE FROM " + self.fq_table + " WHERE " + self.key_col + " IN (SELECT " + self.key_col + " FROM clean_incoming);"
            )
            self.con.execute(
                "INSERT INTO " + self.fq_table + " (" + insert_cols + ") SELECT " + select_expr + " FROM clean_incoming i;"
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

        total_after = int(self.con.execute("SELECT COUNT(*) FROM " + self.fq_table).fetchone()[0])
        inserted_rows = max(incoming_rows - existing_rows, 0)
        updated_rows = max(existing_rows, 0)

        # Sanity: key duplicates should be impossible
        dupes = int(
            self.con.execute(
                "SELECT COUNT(*) FROM (SELECT "
                + self.key_col
                + ", COUNT(*) c FROM "
                + self.fq_table
                + " GROUP BY 1 HAVING c>1)"
            ).fetchone()[0]
        )
        if dupes > 0:
            logging.error("[duckdb_writer] Duplicate keys detected post-upsert: %s", dupes)

        return {
            "incoming_rows": incoming_rows,
            "inserted_rows": inserted_rows,
            "updated_rows": updated_rows,
            "unchanged_rows": 0,
            "total_after": total_after,
        }
