"""Quick exploratory queries against nyc_traffic_2016.duckdb."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_PATHS = [
    Path("nyc_traffic_2016.duckdb"),
    Path("data/nyc_traffic_2016.duckdb"),
]

existing_paths = [p for p in DEFAULT_PATHS if p.exists()]
DB_PATH = max(existing_paths, key=lambda p: p.stat().st_size) if existing_paths else None

if DB_PATH is None:
    raise SystemExit("Database not found. Run data/load_data.py first.")

conn = duckdb.connect(str(DB_PATH))

print("=== Tables ===")
print(conn.execute("SHOW TABLES").fetchdf(), "\n")

def table_exists(name: str) -> bool:
    return (
        conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name.lower()]
        ).fetchone()[0]
        > 0
    )


def describe_table(name: str, limit: int = 5) -> None:
    if not table_exists(name):
        print(f"[WARN] Table {name} not found in {DB_PATH}.")
        return
    print(f"--- {name} schema ---")
    print(conn.execute(f"DESCRIBE {name}").fetchdf())
    print(f"\nSample rows ({limit}):")
    print(conn.execute(f"SELECT * FROM {name} LIMIT {limit}").fetchdf())
    print()

describe_table("taxi_data", limit=3)
describe_table("traffic_2016", limit=3)
describe_table("collisions_2016", limit=3)

print("--- Trip duration stats (minutes) ---")
print(
    conn.execute(
        """
        WITH trips AS (
            SELECT
                TRY_STRPTIME(tpep_pickup_datetime, '%m/%d/%Y %I:%M:%S %p') AS pickup_ts,
                TRY_STRPTIME(tpep_dropoff_datetime, '%m/%d/%Y %I:%M:%S %p') AS dropoff_ts
            FROM taxi_data
            WHERE tpep_pickup_datetime IS NOT NULL AND tpep_dropoff_datetime IS NOT NULL
        )
        SELECT
            AVG(DATE_DIFF('minute', pickup_ts, dropoff_ts)) AS avg_min,
            MIN(DATE_DIFF('minute', pickup_ts, dropoff_ts)) AS min_min,
            MAX(DATE_DIFF('minute', pickup_ts, dropoff_ts)) AS max_min
        FROM trips
        """
    ).fetchdf()
)

print("\n--- OD frequency snapshot ---")
print(
    conn.execute(
        """
        SELECT PULocationID, DOLocationID, COUNT(*) AS trips
        FROM taxi_data
        GROUP BY 1,2
        ORDER BY trips DESC
        LIMIT 10
        """
    ).fetchdf()
)
