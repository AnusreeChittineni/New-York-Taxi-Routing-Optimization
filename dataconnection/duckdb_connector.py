"""Thin DuckDB helpers for NYC Yellow Taxi data access."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

DEFAULT_DB_PATH = "nyc_taxi.duckdb"
PARQUET_GLOB = "yellow_tripdata_2024-*.parquet"


def connect_duckdb(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Create or connect to a DuckDB file."""

    path = Path(db_path)
    conn = duckdb.connect(str(path))
    conn.execute("PRAGMA threads=4;")
    return conn


def _base_query(limit_clause: Optional[int] = None) -> str:
    sql = (
        "SELECT PULocationID, DOLocationID, trip_distance, total_amount, "
        "tpep_pickup_datetime, tpep_dropoff_datetime "
        f"FROM '{PARQUET_GLOB}'"
    )
    if limit_clause:
        sql += f" LIMIT {limit_clause}"
    return sql


def load_trip_data(
    conn: duckdb.DuckDBPyConnection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 100_000,
) -> pd.DataFrame:
    """Load trip slices with optional date filtering."""

    sql = _base_query(limit)
    conditions = []
    if start_date:
        conditions.append(f"tpep_pickup_datetime >= '{start_date}'")
    if end_date:
        conditions.append(f"tpep_dropoff_datetime <= '{end_date}'")
    if conditions:
        sql += f" WHERE {' AND '.join(conditions)}"
    return conn.execute(sql).df()


def sample_hotspot_distribution(
    conn: duckdb.DuckDBPyConnection, n: int = 1000
) -> pd.DataFrame:
    """Return sampled OD hotspot weights from historical trips."""

    sql = (
        "WITH agg AS ("
        " SELECT PULocationID, DOLocationID, COUNT(*) AS trips"
        f"   FROM '{PARQUET_GLOB}'"
        "  GROUP BY 1,2"
        ") "
        "SELECT PULocationID, DOLocationID, trips,"
        " trips / SUM(trips) OVER () AS weight"
        " FROM agg"
        f" ORDER BY trips DESC LIMIT {n}"
    )
    return conn.execute(sql).df()
