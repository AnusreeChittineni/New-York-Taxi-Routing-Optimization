"""Routing helpers for candidate generation and evaluation."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from tqdm import tqdm


def build_nx_from_pyg(data: Data, weight_name: str = "time") -> nx.DiGraph:
    """Convert a PyG Data object into a NetworkX DiGraph and keep edge ids."""

    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    graph = nx.DiGraph()
    for i in range(num_edges):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        graph.add_edge(u, v, edge_id=i, **{weight_name: 1.0})
    return graph


def k_shortest_paths(
    G: nx.DiGraph,
    source: int,
    target: int,
    k: int = 8,
    weight: str = "time",
    diversity_penalty: float | None = None,
) -> List[List[int]]:
    """Compute up to k shortest paths using Yen-like penalty."""

    if source not in G or target not in G:
        return []

    paths = []
    base_weight = nx.get_edge_attributes(G, weight)
    for path_nodes in nx.shortest_simple_paths(G, source, target, weight=weight):
        edge_ids = []
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_ids.append(G[u][v]["edge_id"])
        paths.append(edge_ids)
        if len(paths) >= k:
            break
        if diversity_penalty:
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                base_weight[(u, v)] = base_weight.get((u, v), 1.0) + diversity_penalty
    return paths


def path_time(edge_ids: Iterable[int], edge_pred_times: torch.Tensor) -> float:
    """Return scalar travel time for a path using predicted edge times."""

    idx = torch.tensor(list(edge_ids), dtype=torch.long, device=edge_pred_times.device)
    return float(edge_pred_times[idx].sum().item())


def path_edge_mask(edge_ids: Iterable[int], num_edges: int) -> torch.BoolTensor:
    """Boolean mask over edges for quick aggregation."""

    mask = torch.zeros(num_edges, dtype=torch.bool)
    if not edge_ids:
        return mask
    idx = torch.tensor(list(edge_ids), dtype=torch.long)
    mask[idx] = True
    return mask


def _single_pair_k_paths(
    G: nx.DiGraph, u: int, v: int, k: int, weight: str, diversity_penalty: float | None
) -> Tuple[int, int, List[List[int]]]:
    """Helper for parallel execution."""
    paths = k_shortest_paths(G, u, v, k, weight, diversity_penalty)
    return u, v, paths


def precompute_paths(
    G: nx.DiGraph,
    samples: List[Tuple[int, int, float]],
    k: int = 8,
    weight: str = "time",
    diversity_penalty: float | None = None,
    cache_path: str | None = None,
    n_jobs: int = -1,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Precompute K-shortest paths for all OD pairs in samples, with caching."""
    
    path_cache = {}
    if cache_path and os.path.exists(cache_path):
        print(f"Loading precomputed paths from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                path_cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache ({e}). Starting fresh.")
    
    print(f"Checking checkpoints... Found {len(path_cache)} paths already computed.")

    # Unique OD pairs
    all_od_pairs = list(set((s[0], s[1]) for s in samples))
    missing_pairs = [pair for pair in all_od_pairs if pair not in path_cache]
    
    if not missing_pairs:
        print("All paths already cached.")
        return path_cache

    print(f"Precomputing paths for {len(missing_pairs)} missing OD pairs (k={k})...")
    
    chunk_size = 200
    # Process in chunks to allow checkpointing
    with Parallel(n_jobs=n_jobs) as parallel:
        for i in range(0, len(missing_pairs), chunk_size):
            chunk = missing_pairs[i : i + chunk_size]
            
            results = parallel(
                delayed(_single_pair_k_paths)(G, u, v, k, weight, diversity_penalty)
                for u, v in tqdm(chunk, desc=f"Generating Paths (Chunk {i//chunk_size + 1})", leave=False)
            )
            
            for u, v, p in results:
                path_cache[(u, v)] = p
                
            if cache_path:
                # Atomic write (write to temp then rename) to prevent corruption
                temp_path = cache_path + ".tmp"
                with open(temp_path, "wb") as f:
                    pickle.dump(path_cache, f)
                os.replace(temp_path, cache_path)
                # print(f"Checkpoint saved to {cache_path}")

    return path_cache
