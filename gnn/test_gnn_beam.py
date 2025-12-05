"""Beam-search evaluation for trained GNN models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataconnection.duckdb_connector import connect_duckdb, sample_hotspot_distribution
from gnn.gnn_model import build_model, detect_device, load_model
from gnn.routing import build_nx_from_pyg, k_shortest_paths, path_time


def load_hotspots(db_path: str | None, limit: int = 1000) -> pd.DataFrame:
    if not db_path:
        return pd.DataFrame(columns=["PULocationID", "DOLocationID", "weight"])
    try:
        conn = connect_duckdb(db_path)
        return sample_hotspot_distribution(conn, n=limit)
    except Exception as exc:  # pragma: no cover - fallback
        print(f"[WARN] Hotspot sampling failed: {exc}")
        return pd.DataFrame(columns=["PULocationID", "DOLocationID", "weight"])


def update_graph_weights(G, edge_times: torch.Tensor) -> None:
    for _, _, data in G.edges(data=True):
        data["time"] = float(edge_times[data["edge_id"]])


def evaluate_random_ods(G, edge_times: torch.Tensor, n: int = 500) -> float:
    nodes = list(G.nodes)
    samples = []
    for _ in range(n):
        origin, dest = random.sample(nodes, 2)
        paths = k_shortest_paths(G, origin, dest, k=1, weight="time")
        if not paths:
            continue
        samples.append(path_time(paths[0], edge_times))
    return float(np.mean(samples)) if samples else float("inf")


def evaluate_weighted_ods(
    G, edge_times: torch.Tensor, hotspot_df: pd.DataFrame, n: int = 500
) -> float:
    if hotspot_df.empty:
        return evaluate_random_ods(G, edge_times, n)
    weights = hotspot_df["weight"].to_numpy()
    weights = weights / weights.sum()
    samples = []
    nodes = list(G.nodes)
    for _ in range(n):
        idx = np.random.choice(len(hotspot_df), p=weights)
        origin = int(hotspot_df.iloc[idx]["PULocationID"]) % len(nodes)
        dest = int(hotspot_df.iloc[idx]["DOLocationID"]) % len(nodes)
        if origin == dest:
            dest = (dest + 1) % len(nodes)
        paths = k_shortest_paths(G, origin, dest, k=1, weight="time")
        if not paths:
            continue
        samples.append(path_time(paths[0], edge_times))
    return float(np.mean(samples)) if samples else float("inf")


def evaluate_configuration(
    base_times: torch.Tensor,
    removed_edges: Set[int],
    G,
    hotspot_df: pd.DataFrame,
    random_samples: int,
    weighted_samples: int,
) -> Dict[str, float]:
    edge_times = base_times.clone()
    if removed_edges:
        edge_times[list(removed_edges)] = edge_times.max() * 10.0
    update_graph_weights(G, edge_times)
    avg = evaluate_random_ods(G, edge_times, random_samples)
    weighted = evaluate_weighted_ods(G, edge_times, hotspot_df, weighted_samples)
    return {"avg": avg, "weighted": weighted}


def beam_search(
    base_times: torch.Tensor,
    G,
    hotspot_df: pd.DataFrame,
    beam_width: int,
    steps: int,
    candidate_edges: Sequence[int],
    random_samples: int,
    weighted_samples: int,
) -> List[Tuple[Set[int], Dict[str, float]]]:
    beam: List[Tuple[Set[int], Dict[str, float]]] = []
    initial_metrics = evaluate_configuration(
        base_times, set(), G, hotspot_df, random_samples, weighted_samples
    )
    beam.append((set(), initial_metrics))

    for step in range(steps):
        new_candidates = []
        for removed, metrics in beam:
            remaining = [e for e in candidate_edges if e not in removed]
            for e in remaining:
                new_removed = set(removed)
                new_removed.add(e)
                stats = evaluate_configuration(
                    base_times, new_removed, G, hotspot_df, random_samples, weighted_samples
                )
                new_candidates.append((new_removed, stats))
        if not new_candidates:
            break
        new_candidates.sort(key=lambda item: item[1]["weighted"])
        beam = new_candidates[:beam_width]
        print(f"Beam step {step+1}: best weighted={beam[0][1]['weighted']:.2f}")
    return beam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beam search over road closure scenarios.")
    parser.add_argument("--graph-path", type=str, default="data/manhattan_graph.pt")
    parser.add_argument("--model-path", type=str, default="models/gnn_trained.pth")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--random-samples", type=int, default=200)
    parser.add_argument("--weighted-samples", type=int, default=200)
    parser.add_argument("--result-path", type=str, default="results/beam_search_summary.csv")
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path("results").mkdir(exist_ok=True)

    data = torch.load(args.graph_path)
    device = detect_device()

    model = build_model(data.num_node_features, args.hidden_channels)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    data = data.to(device)

    with torch.no_grad():
        base_edge_times = model(data).cpu()

    G = build_nx_from_pyg(data.cpu())
    update_graph_weights(G, base_edge_times)

    hotspot_df = load_hotspots(args.db_path)
    candidate_edges = list(range(min(args.candidate_count, base_edge_times.numel())))

    beam = beam_search(
        base_edge_times,
        G,
        hotspot_df,
        args.beam_width,
        args.steps,
        candidate_edges,
        args.random_samples,
        args.weighted_samples,
    )

    rows = []
    for removed, metrics in beam:
        rows.append(
            {
                "removed_edges": sorted(list(removed)),
                "avg_travel_time": metrics["avg"],
                "weighted_travel_time": metrics["weighted"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.result_path, index=False)
    print(f"Beam search summary saved to {args.result_path}")


if __name__ == "__main__":
    main()
