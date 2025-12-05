"""Routing helpers for candidate generation and evaluation."""

from __future__ import annotations

from typing import Iterable, List

import networkx as nx
import torch
from torch_geometric.data import Data


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
