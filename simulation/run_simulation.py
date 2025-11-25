"""Configurable, lightweight UE simulation runner."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.simulate_manhattan import (
    MANHATTAN_BBOX,
    build_network,
    generate_od_matrix,
    run_assignment,
)


def apply_scaling(edges, speed_scale: float, capacity_scale: float):
    scaled = edges.copy()
    scaled["speed_kph"] *= speed_scale
    scaled["capacity"] *= capacity_scale
    meters_per_minute = scaled["speed_kph"] * (1000.0 / 60.0)
    scaled["free_flow_time_min"] = scaled["length"] / meters_per_minute.clip(lower=1.0)
    return scaled


def calibrate_to_target(
    edges,
    od_matrix: pd.DataFrame,
    target_travel_time: float,
    bpr_params: Dict[str, float],
    max_iters: int = 4,
    console: Optional[Console] = None,
) -> tuple[pd.DataFrame, float]:
    speed_scale = 1.0
    best_result = None
    for i in range(max_iters):
        scaled_edges = apply_scaling(edges, speed_scale, 1.0)
        result = run_assignment(scaled_edges, od_matrix, bpr_params=bpr_params)
        total = result.total_travel_time
        best_result = result
        if console:
            console.print(
                f"[cyan]Calibration {i+1}/{max_iters}[/cyan]: "
                f"speed_scale={speed_scale:.3f}, total={total:,.0f}"
            )
        if abs(total - target_travel_time) / target_travel_time < 0.02:
            break
        ratio = target_travel_time / max(total, 1.0)
        speed_scale *= ratio ** 0.5
    return apply_scaling(edges, speed_scale, 1.0), best_result.total_travel_time if best_result else np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight UE simulation with tunable parameters.")
    parser.add_argument("--bbox-north", type=float, default=MANHATTAN_BBOX["north"])
    parser.add_argument("--bbox-south", type=float, default=MANHATTAN_BBOX["south"])
    parser.add_argument("--bbox-east", type=float, default=MANHATTAN_BBOX["east"])
    parser.add_argument("--bbox-west", type=float, default=MANHATTAN_BBOX["west"])
    parser.add_argument("--centroids", type=int, default=12)
    parser.add_argument("--od-pairs", type=int, default=150)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--capacity-scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--target-travel-time", type=float, default=None)
    parser.add_argument("--calibration-iters", type=int, default=3)
    parser.add_argument("--output-parquet", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    bbox = {
        "north": args.bbox_north,
        "south": args.bbox_south,
        "east": args.bbox_east,
        "west": args.bbox_west,
    }

    console.print("[bold]Running lightweight UE simulation[/bold]")
    G, nodes_gdf, edges_gdf = build_network(bbox)
    od_matrix = generate_od_matrix(
        graph=G, centroid_count=args.centroids, od_pairs=args.od_pairs, seed=42
    )

    scaled_edges = apply_scaling(edges_gdf, args.speed_scale, args.capacity_scale)
    bpr = {"alpha": args.alpha, "beta": args.beta}

    if args.target_travel_time:
        scaled_edges, _ = calibrate_to_target(
            scaled_edges,
            od_matrix,
            args.target_travel_time,
            bpr_params=bpr,
            max_iters=args.calibration_iters,
            console=console,
        )

    console.print("Running assignment with tuned parameters…")
    results = run_assignment(scaled_edges, od_matrix, bpr_params=bpr)
    console.print(
        f"[green]✓[/green] Total travel time: {results.total_travel_time:,.0f} vehicle-minutes "
        f"(alpha={args.alpha}, beta={args.beta})"
    )

    if args.output_parquet:
        Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
        results.edges.to_parquet(args.output_parquet, index=False)
        console.print(f"[green]✓[/green] Saved edge results to {args.output_parquet}")


if __name__ == "__main__":
    main()
