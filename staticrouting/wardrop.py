"""Placeholder Wardrop static routing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_baseline_results(parquet_path: str | Path = "data/manhattan_baseline.parquet") -> pd.DataFrame:
    """Load previously assigned edges for static routing experiments."""

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"No baseline results at {path}")
    return pd.read_parquet(path)


def compute_system_travel_time(df: pd.DataFrame) -> float:
    """Compute Wardrop-style total travel time from baseline flows."""

    if not {"flow", "travel_time"}.issubset(df.columns):
        raise ValueError("DataFrame must have 'flow' and 'travel_time' columns")
    return float((df["flow"] * df["travel_time"]).sum())
