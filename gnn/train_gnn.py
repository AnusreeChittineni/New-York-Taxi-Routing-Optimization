"""Training script for stochastic candidate-route GNN."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from dataconnection.duckdb_connector import connect_duckdb, load_trip_data
from gnn.exploration_policy import (
    cosine_anneal,
    epsilon_greedy,
    gumbel_softmax_mixture,
    linear_anneal,
    softmax_sample,
)
from gnn.gnn_model import build_model, detect_device, save_model
from gnn.routing import build_nx_from_pyg, k_shortest_paths, path_edge_mask


class TripDataset(Dataset):
    """Simple dataset storing OD pairs and observed durations."""

    def __init__(self, samples: Sequence[Tuple[int, int, float]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, int, float]:
        return self.samples[idx]


def build_trip_samples(
    graph, num_nodes: int, limit: int = 5000, db_path: str | None = None
) -> List[Tuple[int, int, float]]:
    """Load trips from DuckDB; fallback to synthetic samples."""

    samples: List[Tuple[int, int, float]] = []
    if db_path:
        try:
            conn = connect_duckdb(db_path)
            df = load_trip_data(conn, limit=limit)
            if not df.empty:
                trip_durations = (
                    pd.to_datetime(df["tpep_dropoff_datetime"])
                    - pd.to_datetime(df["tpep_pickup_datetime"])
                ).dt.total_seconds() / 60.0
                o_nodes = df["PULocationID"].astype(int) % num_nodes
                d_nodes = df["DOLocationID"].astype(int) % num_nodes
                for o, d, t in zip(o_nodes, d_nodes, trip_durations):
                    if o == d or np.isnan(t):
                        continue
                    samples.append((int(o), int(d), float(max(t, 1.0))))
        except Exception as exc:  # pragma: no cover - fallback
            print(f"[WARN] DuckDB load failed: {exc}. Falling back to synthetic samples.")

    if not samples:
        rng = np.random.default_rng(42)
        for _ in range(limit):
            o, d = rng.choice(num_nodes, size=2, replace=False)
            samples.append((int(o), int(d), float(rng.uniform(5, 30))))
    return samples


def batch_iter(dataset: Sequence, batch_size: int) -> Sequence[Sequence]:
    """Yield mini-batches."""

    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


def select_policy_weights(
    scores: List[float], policy: str, epsilon: float, temperature: float, device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return path weights tensor and monitoring info."""

    policy = policy.lower()
    info = {"policy": policy, "random_pick": 0.0}
    if policy == "epsilon_greedy":
        idx = epsilon_greedy(scores, epsilon)
        weights = torch.zeros(len(scores), device=device)
        weights[idx] = 1.0
        info["random_pick"] = float(epsilon)
        return weights, info
    if policy == "softmax":
        idx = softmax_sample(scores, temperature)
        weights = torch.zeros(len(scores), device=device)
        weights[idx] = 1.0
        entropy = -float(np.log(len(scores)) if len(scores) > 1 else 0.0)
        info["entropy"] = entropy
        return weights, info
    weights = gumbel_softmax_mixture(scores, temperature).to(device)
    return weights, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN with stochastic route sampling.")
    parser.add_argument("--graph-path", type=str, default="data/manhattan_graph.pt")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-path", type=str, default="models/gnn_trained.pth")
    parser.add_argument("--K-CANDS", type=int, default=8)
    parser.add_argument("--POLICY", type=str, default="gumbel_softmax")
    parser.add_argument("--EPS-START", type=float, default=0.30)
    parser.add_argument("--EPS-END", type=float, default=0.02)
    parser.add_argument("--EPS-STEPS", type=int, default=5000)
    parser.add_argument("--TAU-START", type=float, default=2.0)
    parser.add_argument("--TAU-END", type=float, default=0.25)
    parser.add_argument("--TAU-STEPS", type=int, default=5000)
    parser.add_argument("--ENTROPY-BETA", type=float, default=0.001)
    parser.add_argument("--LOSS", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--anneal", type=str, default="linear", choices=["linear", "cosine"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data: torch.Tensor = torch.load(args.graph_path)
    if not hasattr(data, "edge_index"):
        raise ValueError("Graph file must contain edge_index")

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    device = detect_device()

    samples = build_trip_samples(data, num_nodes, db_path=args.db_path)
    dataset = TripDataset(samples)

    model = build_model(data.num_node_features, args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    G = build_nx_from_pyg(data)

    schedule_fn = linear_anneal if args.anneal == "linear" else cosine_anneal
    global_step = 0

    data = data.to(device)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        random.shuffle(dataset.samples)
        progress = tqdm(
            batch_iter(dataset.samples, args.batch_size),
            total=max(1, len(dataset) // args.batch_size),
            desc=f"Epoch {epoch}",
        )
        for batch in progress:
            optimizer.zero_grad()
            edge_pred = model(data)
            batch_losses = []
            entropy_bonus = 0.0
            for origin, dest, obs_time in batch:
                epsilon = schedule_fn(global_step, args.EPS_START, args.EPS_END, args.EPS_STEPS)
                tau = schedule_fn(global_step, args.TAU_START, args.TAU_END, args.TAU_STEPS)

                paths = k_shortest_paths(
                    G, origin, dest, k=args.K_CANDS, weight="time", diversity_penalty=0.1
                )
                if not paths:
                    continue
                scores = [float(path_edge_mask(p, num_edges).float().to(device).mul(edge_pred).sum()) for p in paths]
                weights, info = select_policy_weights(scores, args.POLICY, epsilon, tau, device)

                path_masks = torch.stack([path_edge_mask(p, num_edges).to(device).float() for p in paths])

                if args.POLICY == "gumbel_softmax":
                    mixture_mask = torch.matmul(weights.unsqueeze(0), path_masks).squeeze(0)
                else:
                    idx = weights.argmax().item()
                    mixture_mask = path_masks[idx]
                pred_time = (mixture_mask * edge_pred).sum()

                obs = torch.tensor(obs_time, dtype=torch.float32, device=device)
                if args.LOSS == "mse":
                    loss_val = F.mse_loss(pred_time, obs)
                else:
                    loss_val = F.smooth_l1_loss(pred_time, obs)
                if args.POLICY == "gumbel_softmax":
                    entropy = -torch.sum(weights * torch.log(weights + 1e-8))
                    entropy_bonus += -args.ENTROPY_BETA * entropy
                batch_losses.append(loss_val)
                global_step += 1

            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean() + entropy_bonus
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / max(1, len(dataset) / args.batch_size)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    save_model(model.cpu(), args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
