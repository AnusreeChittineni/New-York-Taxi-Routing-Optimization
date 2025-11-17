"""Standalone RMSE evaluation for trained travel-time GNNs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from dataconnection.duckdb_connector import connect_duckdb, load_trip_data
from gnn.gnn_model import build_model, detect_device
from gnn.routing import build_nx_from_pyg, k_shortest_paths, path_edge_mask

TripSample = Tuple[int, int, float]


def load_samples_from_parquet(path: Path, num_nodes: int) -> List[TripSample]:
    df = pd.read_parquet(path)
    return [
        (int(row.origin) % num_nodes, int(row.destination) % num_nodes, float(row.travel_min))
        for row in df.itertuples(index=False)
    ]


def fetch_samples_from_duckdb(num_nodes: int, limit: int, db_path: str | None) -> List[TripSample]:
    conn = connect_duckdb(db_path)
    df = load_trip_data(conn, limit=limit)
    pickups = pd.to_datetime(df["pickup_ts"])
    dropoffs = pd.to_datetime(df["dropoff_ts"])
    durations = (dropoffs - pickups).dt.total_seconds() / 60.0
    samples = []
    for pu, do, dur in zip(df["PULocationID"], df["DOLocationID"], durations):
        if pd.isna(pu) or pd.isna(do) or pd.isna(dur) or dur <= 0:
            continue
        samples.append((int(pu) % num_nodes, int(do) % num_nodes, float(dur)))
    return samples


def compute_rmse(model, data, G, samples, num_edges, device, k_candidates=3) -> float:
    model.eval()
    with torch.no_grad():
        edge_pred = model(data).detach()
    cpu_vals = edge_pred.cpu().numpy()
    for _, _, attr in G.edges(data=True):
        attr["time"] = float(cpu_vals[attr["edge_id"]])

    sq_errors = []
    for origin, dest, obs in samples:
        paths = k_shortest_paths(G, origin, dest, k=k_candidates, weight="time")
        if not paths:
            continue
        times = []
        for p in paths:
            mask = path_edge_mask(p, num_edges).to(device).float()
            times.append(float((mask * edge_pred).sum().item()))
        pred_time = min(times)
        sq_errors.append((pred_time - obs) ** 2)
    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained GNN RMSE on held-out trips.")
    parser.add_argument("--graph-path", type=str, default="data/manhattan_graph.pt")
    parser.add_argument("--model-path", type=str, default="models/gnn_trained.pth")
    parser.add_argument("--val-samples-path", type=str, default="data/trip_samples_val.parquet")
    parser.add_argument("--db-path", type=str, default="data/nyc_traffic_2016.duckdb")
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--k-cands", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.graph_path)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    device = detect_device()

    if Path(args.val_samples_path).exists():
        print(f"Loading validation samples from {args.val_samples_path}")
        samples = load_samples_from_parquet(Path(args.val_samples_path), num_nodes)
    else:
        print("Validation samples parquet not found; querying DuckDB directly.")
        samples = fetch_samples_from_duckdb(num_nodes, args.sample_count, args.db_path)

    if len(samples) > args.sample_count:
        samples = random.sample(samples, args.sample_count)

    model = build_model(data.num_node_features, args.hidden_channels)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    data = data.to(device)

    rmse = compute_rmse(model, data, build_nx_from_pyg(data.cpu()), samples, num_edges, device, args.k_cands)
    print(f"RMSE over {len(samples)} samples: {rmse:.3f} minutes")


if __name__ == "__main__":
    main()
