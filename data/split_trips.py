"""Utility to build train/validation trip samples from DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataconnection.duckdb_connector import connect_duckdb


def fetch_trip_samples(limit: int | None = 200_000) -> pd.DataFrame:
    conn = connect_duckdb()
    limit_clause = f" LIMIT {limit}" if limit else ""
    sql = f"""
        WITH raw AS (
            SELECT
                CAST(PULocationID AS INTEGER) AS origin,
                CAST(DOLocationID AS INTEGER) AS destination,
                TRY_STRPTIME(tpep_pickup_datetime, '%m/%d/%Y %I:%M:%S %p') AS pickup_ts,
                TRY_STRPTIME(tpep_dropoff_datetime, '%m/%d/%Y %I:%M:%S %p') AS dropoff_ts
            FROM taxi_data
        ),
        trips AS (
            SELECT
                origin,
                destination,
                DATE_DIFF('minute', pickup_ts, dropoff_ts) AS travel_min
            FROM raw
            WHERE pickup_ts IS NOT NULL
              AND dropoff_ts IS NOT NULL
              AND origin IS NOT NULL
              AND destination IS NOT NULL
        )
        SELECT *
        FROM trips
        WHERE travel_min BETWEEN 1 AND 240
        {limit_clause}
    """
    df = conn.execute(sql).fetchdf()
    return df.dropna().astype({"origin": "int64", "destination": "int64", "travel_min": "float64"})


def split_dataset(df: pd.DataFrame, val_ratio: float, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < val_ratio
    val_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DuckDB trips into train/val parquet files.")
    parser.add_argument("--limit", type=int, default=200_000, help="Max rows to sample from taxi_data")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    df = fetch_trip_samples(limit=args.limit)
    train_df, val_df = split_dataset(df, args.val_ratio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "trip_samples_train.parquet"
    val_path = output_dir / "trip_samples_val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"Saved {len(train_df):,} train samples -> {train_path}")
    print(f"Saved {len(val_df):,} validation samples -> {val_path}")


if __name__ == "__main__":
    main()
