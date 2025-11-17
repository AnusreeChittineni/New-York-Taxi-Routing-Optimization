#!/usr/bin/env python3
"""Convenience runner for major workflows."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_command(cmd: list[str], desc: str) -> None:
    print(f"[RUN] {desc}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_baseline_sim() -> None:
    run_command(["python", "simulation/simulate_manhattan.py"], "Baseline Manhattan UE simulation")


def train_gnn_default() -> None:
    run_command(["python", "gnn/train_gnn.py"], "Train GNN (default SCRT config)")


def test_gnn_rmse() -> None:
    run_command(["python", "gnn/test_gnn_rmse.py"], "Evaluate GNN RMSE on validation set")


def split_trip_data() -> None:
    run_command(["python", "data/split_trips.py"], "Build train/validation trip samples from DuckDB")


def beam_search_eval() -> None:
    run_command(["python", "gnn/test_gnn_beam.py"], "Beam-search evaluation")


def wardrop_summary() -> None:
    run_command(["python", "-c", "from staticrouting.wardrop import load_baseline_results, compute_system_travel_time; "
        "print(compute_system_travel_time(load_baseline_results()))"], "Wardrop summary")


if __name__ == "__main__":
    # Uncomment the routines you want to run:
    # run_baseline_sim()
    # split_trip_data()
    # train_gnn_default()
    # test_gnn_rmse()
    # beam_search_eval()
    # wardrop_summary()
    pass
