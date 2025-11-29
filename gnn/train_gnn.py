"""Training script for stochastic candidate-route GNN with RMSE reporting."""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import time
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
from tqdm import tqdm

from dataconnection.duckdb_connector import connect_duckdb, load_trip_data
from gnn.exploration_policy import (
    cosine_anneal,
    epsilon_greedy,
    gumbel_softmax_mixture,
    linear_anneal,
    softmax_sample,
)
from gnn.gnn_model import build_model, detect_device, load_model, save_model
from gnn.routing import build_nx_from_pyg, k_shortest_paths, path_edge_mask, precompute_paths


TripSample = Tuple[int, int, float]


def batch_iter(dataset: Sequence[TripSample], batch_size: int) -> Sequence[Sequence[TripSample]]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


def load_samples_from_parquet(path: Path, num_nodes: int) -> List[TripSample]:
    df = pd.read_parquet(path)
    return [
        (int(row.origin) % num_nodes, int(row.destination) % num_nodes, float(row.travel_min))
        for row in df.itertuples(index=False)
    ]


def build_trip_samples(num_nodes: int, limit: int, db_path: str | None) -> List[TripSample]:
    samples: List[TripSample] = []
    try:
        conn = connect_duckdb(db_path)
        df = load_trip_data(conn, limit=limit)
        if not df.empty:
            pickups = pd.to_datetime(df["pickup_ts"])
            dropoffs = pd.to_datetime(df["dropoff_ts"])
            durations = (dropoffs - pickups).dt.total_seconds() / 60.0
            for pu, do, dur in zip(df["PULocationID"], df["DOLocationID"], durations):
                if pd.isna(pu) or pd.isna(do) or pd.isna(dur) or dur <= 0:
                    continue
                samples.append((int(pu) % num_nodes, int(do) % num_nodes, float(dur)))
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] DuckDB load failed ({exc}); using synthetic fallback.")

    if not samples:
        rng = np.random.default_rng(42)
        for _ in range(limit):
            o, d = rng.choice(num_nodes, size=2, replace=False)
            samples.append((int(o), int(d), float(rng.uniform(5, 30))))
    return samples


def train_val_split(samples: List[TripSample], val_ratio: float, seed: int = 42) -> tuple[List[TripSample], List[TripSample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    cutoff = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:cutoff], shuffled[cutoff:]


def limit_samples(samples: List[TripSample], count: int, seed: int = 42) -> List[TripSample]:
    if count <= 0 or count >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, count)


def select_policy_weights(
    scores: List[float], policy: str, epsilon: float, temperature: float, device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    info = {"policy": policy, "random_pick": 0.0}
    policy = policy.lower()
    if policy == "greedy":
        idx = int(np.argmin(scores))
        weights = torch.zeros(len(scores), device=device)
        weights[idx] = 1.0
        return weights, info
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
        info["entropy"] = -float(np.log(len(scores)) if len(scores) > 1 else 0.0)
        return weights, info
    weights = gumbel_softmax_mixture(scores, temperature).to(device)
    return weights, info


def update_graph_costs(G, edge_times: torch.Tensor) -> None:
    cpu_vals = edge_times.detach().cpu().numpy()
    for _, _, data in G.edges(data=True):
        data["time"] = float(cpu_vals[data["edge_id"]])


def evaluate_rmse(
    model: torch.nn.Module,
    data,
    G,
    samples: Sequence[TripSample],
    num_edges: int,
    device: torch.device,
    max_eval: int = 500,
    k_candidates: int = 3,
) -> float:
    model.eval()
    with torch.no_grad():
        edge_pred = model(data).detach()
    update_graph_costs(G, edge_pred)

    subset = samples if len(samples) <= max_eval else random.sample(samples, max_eval)
    sq_errors = []
    for origin, dest, obs in subset:
        paths = k_shortest_paths(G, origin, dest, k=k_candidates, weight="time")
        if not paths:
            continue
        times = []
        for p in paths:
            mask = path_edge_mask(p, num_edges).to(device).float()
            times.append(float((mask * edge_pred).sum().item()))
        pred_time = min(times)
        sq_errors.append((pred_time - obs) ** 2)

    return math.sqrt(float(np.mean(sq_errors))) if sq_errors else float("nan")


def find_latest_checkpoint(model_path: str) -> Tuple[str | None, int]:
    """Finds the latest checkpoint based on the model path pattern."""
    base_name = os.path.splitext(model_path)[0]
    extension = os.path.splitext(model_path)[1]
    # Pattern: base_name + "_epoch_" + N + extension
    search_pattern = f"{base_name}_epoch_*{extension}"
    files = glob.glob(search_pattern)
    
    if not files:
        return None, 0
        
    latest_epoch = 0
    latest_file = None
    
    for f in files:
        try:
            # Extract epoch number
            # f is like "models/gnn_trained_epoch_5.pth"
            # remove extension -> "models/gnn_trained_epoch_5"
            # split by "_" -> [..., "epoch", "5"]
            parts = os.path.splitext(f)[0].split("_")
            if "epoch" in parts:
                epoch_idx = parts.index("epoch")
                if epoch_idx + 1 < len(parts):
                    epoch_num = int(parts[epoch_idx + 1])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_file = f
        except ValueError:
            continue
            
    return latest_file, latest_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN with candidate-route sampling.")
    parser.add_argument("--graph-path", type=str, default="data/nyc_graph.pt")
    parser.add_argument("--db-path", type=str, default="data/nyc_traffic_2016.duckdb")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-path", type=str, default="models/gnn_trained.pth")
    parser.add_argument("--trip-limit", type=int, default=50_000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-sample-count", type=int, default=10_000)
    parser.add_argument("--val-sample-count", type=int, default=1_000)
    parser.add_argument("--train-samples-path", type=str, default=None)
    parser.add_argument("--val-samples-path", type=str, default=None)
    parser.add_argument("--K-CANDS", type=int, default=8)
    parser.add_argument("--POLICY", type=str, default="gumbel_softmax")
    parser.add_argument("--policy-mode", type=str, choices=["easy", "complex"], default="complex")
    parser.add_argument("--EPS-START", type=float, default=0.30)
    parser.add_argument("--EPS-END", type=float, default=0.02)
    parser.add_argument("--EPS-STEPS", type=int, default=5000)
    parser.add_argument("--TAU-START", type=float, default=2.0)
    parser.add_argument("--TAU-END", type=float, default=0.25)
    parser.add_argument("--TAU-STEPS", type=int, default=5000)
    parser.add_argument("--ENTROPY-BETA", type=float, default=0.001)
    parser.add_argument("--LOSS", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--anneal", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--rmse-eval-samples", type=int, default=500)
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory to store path caches")
    parser.add_argument("--stats-path", type=str, default="training_stats.csv", help="Path to save training statistics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data = torch.load(args.graph_path, weights_only=False)
    if not hasattr(data, "edge_index"):
        raise ValueError("Graph file must contain edge_index")

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    device = detect_device()

    if args.train_samples_path and args.val_samples_path:
        train_samples = load_samples_from_parquet(Path(args.train_samples_path), num_nodes)
        val_samples = load_samples_from_parquet(Path(args.val_samples_path), num_nodes)
    else:
        raw_samples = build_trip_samples(num_nodes, args.trip_limit, args.db_path)
        train_samples, val_samples = train_val_split(raw_samples, args.val_ratio)

    train_samples = limit_samples(train_samples, args.train_sample_count)
    val_samples = limit_samples(val_samples, args.val_sample_count)

    training_policy = args.POLICY if args.policy_mode == "complex" else "greedy"
    k_candidates = args.K_CANDS if args.policy_mode == "complex" else min(3, args.K_CANDS)
    diversity_penalty = 0.1 if args.policy_mode == "complex" else 0.0

    edge_dim = data.edge_attr.size(-1) if getattr(data, "edge_attr", None) is not None else 0
    model = build_model(data.num_node_features, args.hidden_channels, edge_attr_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    G = build_nx_from_pyg(data.cpu())
    schedule_fn = linear_anneal if args.anneal == "linear" else cosine_anneal
    global_step = 0

    # Check for checkpoints
    start_epoch = 1
    latest_ckpt, latest_epoch = find_latest_checkpoint(args.model_path)
    if latest_ckpt:
        print(f"Found checkpoint: {latest_ckpt}. Resuming from epoch {latest_epoch + 1}")
        model = load_model(latest_ckpt, data.num_node_features, args.hidden_channels, edge_attr_dim=edge_dim).to(device)
        # Note: We are not loading optimizer state here for simplicity, but ideally we should.
        # Assuming the user just wants to continue training weights.
        # If optimizer state is needed, save_model/load_model needs to handle it.
        start_epoch = latest_epoch + 1
        # Approximate global step? Or just reset. 
        # For annealing, global_step matters. 
        # Let's estimate global_step based on epochs * batches
        batches_per_epoch = len(train_samples) // args.batch_size
        global_step = latest_epoch * batches_per_epoch

    # Initialize stats file
    stats_header = "epoch,duration_sec,avg_loss,train_rmse,val_rmse\n"
    if not os.path.exists(args.stats_path):
        with open(args.stats_path, "w") as f:
            f.write(stats_header)

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_file = os.path.join(args.cache_dir, f"paths_k{k_candidates}_p{args.policy_mode}.pkl")
    
    # Precompute paths using static weights (distance/free-flow)
    # Note: We use the initial graph state. If edge weights evolve significantly, 
    # one might want to re-compute periodically, but for speed we do it once.
    path_cache = precompute_paths(
        G, 
        train_samples + val_samples, 
        k=k_candidates, 
        weight="time", # Assumes 'time' is in G from build_nx_from_pyg
        diversity_penalty=diversity_penalty,
        cache_path=cache_file
    )

    data = data.to(device)

    data = data.to(device)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_losses = []
        random.shuffle(train_samples)
        batches = list(batch_iter(train_samples, args.batch_size))
        progress = tqdm(batches, desc=f"Epoch {epoch}")

        for batch in progress:
            model.train()
            optimizer.zero_grad()
            edge_pred = model(data)
            # update_graph_costs(G, edge_pred) # No longer updating graph for routing per step!
            
            batch_losses = []
            entropy_bonus = 0.0

            for origin, dest, obs_time in batch:
                epsilon = schedule_fn(global_step, args.EPS_START, args.EPS_END, args.EPS_STEPS)
                tau = schedule_fn(global_step, args.TAU_START, args.TAU_END, args.TAU_STEPS)
                
                # Retrieve precomputed paths
                paths = path_cache.get((origin, dest), [])
                if not paths:
                    continue

                path_masks = torch.stack([path_edge_mask(p, num_edges).to(device).float() for p in paths])
                scores = [float((mask * edge_pred).sum().item()) for mask in path_masks]
                weights, _ = select_policy_weights(scores, training_policy, epsilon, tau, device)

                if training_policy == "gumbel_softmax":
                    mixture_mask = torch.matmul(weights.unsqueeze(0), path_masks).squeeze(0)
                else:
                    mixture_mask = path_masks[weights.argmax().item()]

                pred_time = (mixture_mask * edge_pred).sum()
                obs = torch.tensor(obs_time, dtype=torch.float32, device=device)
                loss_val = F.mse_loss(pred_time, obs) if args.LOSS == "mse" else F.smooth_l1_loss(pred_time, obs)
                if training_policy == "gumbel_softmax":
                    entropy = -torch.sum(weights * torch.log(weights + 1e-8))
                    entropy_bonus -= args.ENTROPY_BETA * entropy
                batch_losses.append(loss_val)
                global_step += 1

            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean() + entropy_bonus
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            progress.set_postfix({"loss": loss.item()})

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        
        print(f"Evaluating Train RMSE ({args.rmse_eval_samples} samples)...")
        train_rmse = evaluate_rmse(
            model, data, G, train_samples, num_edges, device, args.rmse_eval_samples, k_candidates
        )
        
        print(f"Evaluating Validation RMSE ({args.rmse_eval_samples} samples)...")
        val_rmse = evaluate_rmse(
            model, data, G, val_samples, num_edges, device, args.rmse_eval_samples, k_candidates
        )
        print(
            f"Epoch {epoch}: avg loss={avg_loss:.4f} | train RMSE={train_rmse:.3f} min | "
            f"val RMSE={val_rmse:.3f} min (policy_mode={args.policy_mode})"
        )

        # Save checkpoint
        base, ext = os.path.splitext(args.model_path)
        ckpt_path = f"{base}_epoch_{epoch}{ext}"
        print(f"Saving checkpoint to {ckpt_path}...")
        save_model(model.cpu(), ckpt_path)
        model.to(device) # Move back to device for next epoch
        
        # Log stats
        duration = time.time() - epoch_start_time
        print(f"Logging stats to {args.stats_path}...")
        with open(args.stats_path, "a") as f:
            f.write(f"{epoch},{duration:.2f},{avg_loss:.6f},{train_rmse:.6f},{val_rmse:.6f}\n")

    # Save final model as well (copy of last checkpoint effectively, but good to have the main name)
    print(f"Saving final model to {args.model_path}...")
    save_model(model.cpu(), args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
