"""Compute average trip durations (minutes) from the DuckDB taxi data."""

from __future__ import annotations

from pathlib import Path

import duckdb

DB_CANDIDATES = [
    Path("data/nyc_traffic_2016.duckdb"),
    Path("nyc_traffic_2016.duckdb"),
]


def locate_db() -> Path:
    for path in DB_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("nyc_traffic_2016.duckdb not found in data/ or repo root.")


def main() -> None:
    db_path = locate_db()
    conn = duckdb.connect(str(db_path))
    print(f"Connected to {db_path}")
    result = conn.execute(
        """
        WITH trips AS (
            SELECT
                DATE_DIFF(
                    'minute',
                    TRY_STRPTIME(tpep_pickup_datetime, '%m/%d/%Y %I:%M:%S %p'),
                    TRY_STRPTIME(tpep_dropoff_datetime, '%m/%d/%Y %I:%M:%S %p')
                ) AS travel_min
            FROM taxi_data
            WHERE tpep_pickup_datetime IS NOT NULL
              AND tpep_dropoff_datetime IS NOT NULL
        )
        SELECT
            AVG(travel_min) AS avg_minutes,
            MIN(travel_min) AS min_minutes,
            MAX(travel_min) AS max_minutes,
            COUNT(*) AS sample_size
        FROM trips
        WHERE travel_min BETWEEN 1 AND 240
        """
    ).fetchdf()
    print(result)


if __name__ == "__main__":
    main()
