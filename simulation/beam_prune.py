"""Beam search edge pruning for UE simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rich.console import Console

from simulation.simulate_manhattan import MANHATTAN_BBOX, build_network, generate_od_matrix, run_assignment


def select_candidate_edges(edges: pd.DataFrame, limit: Optional[int]) -> List[int]:
    if limit is None or limit <= 0 or limit >= edges.shape[0]:
        return edges["link_id"].tolist()
    ranked = edges.sort_values("length", ascending=False).head(limit)
    return ranked["link_id"].tolist()


def run_with_removals(
    edges: pd.DataFrame,
    removals: Set[int],
    od_matrix: pd.DataFrame,
    bpr_params: Dict[str, float],
) -> Tuple[float, pd.DataFrame]:
    pruned = edges[~edges["link_id"].isin(removals)].copy()
    result = run_assignment(pruned, od_matrix, bpr_params=bpr_params)
    return result.total_travel_time, result.edges


def beam_search_pruning(
    edges: pd.DataFrame,
    od_matrix: pd.DataFrame,
    bpr_params: Dict[str, float],
    beam_width: int,
    max_steps: Optional[int],
    candidate_limit: Optional[int],
    console: Console,
) -> List[Tuple[Set[int], float]]:
    candidate_pool = select_candidate_edges(edges, candidate_limit)
    beam: List[Tuple[Set[int], float]] = []
    base_tt, _ = run_with_removals(edges, set(), od_matrix, bpr_params)
    beam.append((set(), base_tt))
    step = 0
    limit_steps = max_steps if max_steps and max_steps > 0 else len(candidate_pool)

    while step < limit_steps:
        step += 1
        new_candidates = []
        for removed_set, _ in beam:
            remaining = [eid for eid in candidate_pool if eid not in removed_set]
            if not remaining:
                continue
            for edge_id in remaining:
                updated = set(removed_set)
                updated.add(edge_id)
                total_tt, _ = run_with_removals(edges, updated, od_matrix, bpr_params)
                new_candidates.append((updated, total_tt))
        if not new_candidates:
            break
        new_candidates.sort(key=lambda x: x[1])
        beam = new_candidates[:beam_width]
        console.print(
            f"[cyan]Beam step {step}[/cyan]: best total time={beam[0][1]:,.0f} (removed {len(beam[0][0])} edges)"
        )
        if any(len(removed) == len(candidate_pool) for removed, _ in beam):
            console.print("[yellow]All candidate edges removed. Stopping search.[/yellow]")
            break
    return beam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beam search edge pruning for UE simulation.")
    parser.add_argument("--bbox-north", type=float, default=MANHATTAN_BBOX["north"])
    parser.add_argument("--bbox-south", type=float, default=MANHATTAN_BBOX["south"])
    parser.add_argument("--bbox-east", type=float, default=MANHATTAN_BBOX["east"])
    parser.add_argument("--bbox-west", type=float, default=MANHATTAN_BBOX["west"])
    parser.add_argument("--centroids", type=int, default=12)
    parser.add_argument("--od-pairs", type=int, default=120)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5, help="Set <=0 to keep pruning until no edges remain.")
    parser.add_argument("--candidate-limit", type=int, default=50, help="Limit edges considered for pruning.")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--result-csv", type=str, default="results/beam_pruning.csv")
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
    console.print("[bold]Running beam search edge pruning[/bold]")
    G, nodes_gdf, edges_gdf = build_network(bbox)
    od_matrix = generate_od_matrix(G, centroid_count=args.centroids, od_pairs=args.od_pairs)
    bpr = {"alpha": args.alpha, "beta": args.beta}
    beams = beam_search_pruning(
        edges_gdf,
        od_matrix,
        bpr,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        candidate_limit=args.candidate_limit,
        console=console,
    )
    rows = []
    for removed, total in beams:
        rows.append({"removed_edges": sorted(removed), "total_travel_time": total})

    result_path = Path(args.result_csv)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(result_path, index=False)
    console.print(f"[green]✓[/green] Beam search summary saved to {result_path}")


if __name__ == "__main__":
    main()
