"""Thin DuckDB helpers for NYC Yellow Taxi data access."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

DEFAULT_DB_CANDIDATES = [
    Path("data/nyc_traffic_2016.duckdb"),
    Path("nyc_traffic_2016.duckdb"),
]


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path:
        return Path(db_path)
    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    # fall back to first path even if missing (duckdb will create)
    return DEFAULT_DB_CANDIDATES[0]


def connect_duckdb(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Create or connect to a DuckDB file."""

    path = _resolve_db_path(db_path)
    conn = duckdb.connect(str(path))
    conn.execute("PRAGMA threads=4;")
    return conn


def load_trip_data(
    conn: duckdb.DuckDBPyConnection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 100_000,
) -> pd.DataFrame:
    """Load trip slices with optional date filtering."""

    conditions = ["pickup_ts IS NOT NULL", "dropoff_ts IS NOT NULL"]
    if start_date:
        conditions.append(f"pickup_ts >= TIMESTAMP '{start_date}'")
    if end_date:
        conditions.append(f"dropoff_ts <= TIMESTAMP '{end_date}'")
    where_clause = " AND ".join(conditions)
    limit_clause = f" LIMIT {limit}" if limit else ""
    sql = f"""
        WITH trips AS (
            SELECT
                PULocationID,
                DOLocationID,
                trip_distance,
                total_amount,
                TRY_STRPTIME(tpep_pickup_datetime, '%m/%d/%Y %I:%M:%S %p') AS pickup_ts,
                TRY_STRPTIME(tpep_dropoff_datetime, '%m/%d/%Y %I:%M:%S %p') AS dropoff_ts
            FROM taxi_data
        )
        SELECT * FROM trips
        WHERE {where_clause}
        {limit_clause}
    """
    return conn.execute(sql).df()


def sample_hotspot_distribution(
    conn: duckdb.DuckDBPyConnection, n: int = 1000
) -> pd.DataFrame:
    """Return sampled OD hotspot weights from historical trips."""

    sql = (
        "WITH agg AS ("
        " SELECT PULocationID, DOLocationID, COUNT(*) AS trips"
        "   FROM taxi_data"
        "  GROUP BY 1,2"
        ") "
        "SELECT PULocationID, DOLocationID, trips,"
        " trips / SUM(trips) OVER () AS weight"
        " FROM agg"
        f" ORDER BY trips DESC LIMIT {n}"
    )
    return conn.execute(sql).df()
