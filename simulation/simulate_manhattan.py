#!/usr/bin/env python3
"""Baseline Manhattan UE traffic assignment using OSMnx + AequilibraE."""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from rich.console import Console
from rich.table import Table

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
DATA_DIR = Path("data")
BASELINE_PARQUET = DATA_DIR / "manhattan_baseline.parquet"
NODE_PARQUET = DATA_DIR / "manhattan_nodes.parquet"
OD_PARQUET = DATA_DIR / "manhattan_od.parquet"
PLOT_PATH = Path("manhattan_baseline.png")

MANHATTAN_BBOX = {
    "north": 40.882214,
    "south": 40.680396,
    "east": -73.907000,
    "west": -74.047285,
}

DEFAULT_SPEED_KPH = 35.0
SPEED_LOOKUP: Dict[str, float] = {
    "motorway": 90.0,
    "trunk": 80.0,
    "primary": 60.0,
    "secondary": 45.0,
    "tertiary": 40.0,
    "residential": 30.0,
    "living_street": 25.0,
    "service": 20.0,
    "unclassified": 30.0,
}

CAPACITY_LOOKUP: Dict[str, int] = {
    "motorway": 4200,
    "trunk": 3600,
    "primary": 3200,
    "secondary": 2600,
    "tertiary": 2000,
    "residential": 900,
    "living_street": 600,
    "service": 400,
    "unclassified": 900,
}

NUM_CENTROIDS = 18
OD_PAIRS = 220
COMPUTE_DEVICE = "cpu"


@dataclass
class AssignmentOutputs:
    """Container for assignment outputs."""

    edges: gpd.GeoDataFrame
    total_travel_time: float
    convergence_report: List[dict]


def set_global_state() -> None:
    """Configure logging, caching, and randomness."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("ignore", category=UserWarning)
    ox.settings.use_cache = True
    ox.settings.log_console = False


def normalize_highway(value: Iterable[str] | str) -> str:
    """Return a scalar highway class."""

    if isinstance(value, (list, tuple, set)):
        return next(iter(value))
    return value or "unclassified"


def build_network(
    bbox: Dict[str, float],
) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Download and prep the Manhattan road network.

    Returns the raw MultiDiGraph, nodes GeoDataFrame, and enriched edge GeoDataFrame.
    """

    bbox_tuple = (bbox["north"], bbox["south"], bbox["east"], bbox["west"])
    G = ox.graph_from_bbox(bbox_tuple, network_type="drive", simplify=True)
    G = ox.utils_graph.get_largest_component(G, strongly=True)
    G = ox.add_edge_speeds(G)  # adds speed_kph based on highway types

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    edges_gdf = edges_gdf.reset_index(drop=False).copy()
    edges_gdf["link_id"] = np.arange(edges_gdf.shape[0], dtype=np.int64)
    edges_gdf["highway"] = edges_gdf["highway"].apply(normalize_highway)
    edges_gdf["speed_kph"] = edges_gdf["speed_kph"].fillna(
        edges_gdf["highway"].map(SPEED_LOOKUP)
    )
    edges_gdf["speed_kph"] = edges_gdf["speed_kph"].fillna(DEFAULT_SPEED_KPH)

    # Estimate free-flow travel time in minutes
    meters_per_minute = edges_gdf["speed_kph"] * (1000.0 / 60.0)
    edges_gdf["free_flow_time_min"] = edges_gdf["length"] / meters_per_minute.clip(lower=1.0)

    # Heuristic capacities
    edges_gdf["capacity"] = edges_gdf["highway"].map(CAPACITY_LOOKUP).fillna(1200)

    nodes_gdf = nodes_gdf.reset_index().rename(columns={"osmid": "node_id"})
    nodes_gdf["node_id"] = nodes_gdf["node_id"].astype(np.int64)

    return G, nodes_gdf, edges_gdf


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great-circle distance in kilometers."""

    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return 6371.0 * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def generate_od_matrix(
    graph: nx.MultiDiGraph,
    centroid_count: int = NUM_CENTROIDS,
    od_pairs: int = OD_PAIRS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create a synthetic OD demand matrix with centroid nodes drawn from the network.
    """

    rng = np.random.default_rng(seed)
    candidate_nodes = list(graph.nodes)
    centroid_count = min(len(candidate_nodes), centroid_count)
    centroid_nodes = rng.choice(candidate_nodes, size=centroid_count, replace=False)

    coords = {
        nid: (graph.nodes[nid]["y"], graph.nodes[nid]["x"]) for nid in centroid_nodes
    }

    od_matrix = np.zeros((centroid_count, centroid_count), dtype=np.float64)
    for _ in range(od_pairs):
        origin_idx, dest_idx = rng.choice(centroid_count, size=2, replace=False)
        origin = centroid_nodes[origin_idx]
        dest = centroid_nodes[dest_idx]
        dist = haversine_km(*coords[origin], *coords[dest])
        base = rng.integers(50, 300)
        demand = float(base * (1.0 + dist / 5.0))
        od_matrix[origin_idx, dest_idx] += demand

    index = pd.Index(centroid_nodes.astype(np.int64), name="node_id")
    return pd.DataFrame(od_matrix, index=index, columns=index)


def build_graph_dataframe(edges: gpd.GeoDataFrame) -> pd.DataFrame:
    """Translate GeoDataFrame into the schema expected by AequilibraE Graph."""

    network_df = pd.DataFrame(
        {
            "link_id": edges["link_id"].astype(np.int64),
            "a_node": edges["u"].astype(np.int64),
            "b_node": edges["v"].astype(np.int64),
            "direction": 1,
            "length": edges["length"].astype(float),
            "capacity": edges["capacity"].astype(float),
            "free_flow_time": edges["free_flow_time_min"].astype(float),
        }
    )
    return network_df


def prepare_aequilibrae_objects(
    edges: gpd.GeoDataFrame, od_matrix: pd.DataFrame, matrix_name: str = "demand"
) -> Tuple[Graph, AequilibraeMatrix, str]:
    """Create the Graph and matrix objects required for assignment."""

    network_df = build_graph_dataframe(edges)
    graph = Graph()
    graph.mode = "auto"
    graph.network = network_df
    centroids = od_matrix.index.to_numpy(dtype=np.int64)
    graph.prepare_graph(centroids=centroids)
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(True)

    demand_matrix = AequilibraeMatrix()
    demand_matrix.create_empty(
        zones=len(centroids), matrix_names=[matrix_name], memory_only=True
    )
    demand_matrix.index[:] = centroids
    demand_matrix.matrix[matrix_name][:, :] = od_matrix.to_numpy(dtype=np.float64)
    demand_matrix.computational_view([matrix_name])

    return graph, demand_matrix, matrix_name


def run_assignment(
    edges: gpd.GeoDataFrame, od_matrix: pd.DataFrame, console: Optional[Console] = None
) -> AssignmentOutputs:
    """Execute a UE traffic assignment using AequilibraE."""

    graph, demand_matrix, matrix_name = prepare_aequilibrae_objects(edges, od_matrix)
    traffic_class = TrafficClass("autos", graph, demand_matrix)

    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.set_algorithm("bfw")
    assignment.max_iter = 40
    assignment.rgap_target = 1e-4
    assignment.execute()

    result_df = assignment.results().reset_index()
    demand_field = f"{matrix_name}_ab"
    time_field = "Congested_Time_AB"

    merged = edges.merge(result_df, on="link_id", how="left")
    merged[demand_field] = merged[demand_field].fillna(0.0)
    merged[time_field] = merged[time_field].fillna(merged["free_flow_time_min"])
    merged["flow"] = merged[demand_field]
    merged["travel_time"] = merged[time_field]

    total_travel_time = float((merged["flow"] * merged["travel_time"]).sum())

    if console:
        report = assignment.assignment.convergence_report or []
        final_gap = report[-1]["rgap"] if report else float("nan")
        console.print(
            f"[green]✓[/green] Ran User Equilibrium assignment "
            f"(iterations={len(report)}, final rgap={final_gap:.6f})"
        )

    demand_matrix.close()

    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=edges.crs)
    return AssignmentOutputs(
        edges=merged_gdf,
        total_travel_time=total_travel_time,
        convergence_report=assignment.assignment.convergence_report,
    )


def save_results(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    od_matrix: pd.DataFrame,
) -> None:
    """Persist edge/node/OD features for downstream ML workflows."""

    DATA_DIR.mkdir(exist_ok=True)

    edge_export = edges[
        [
            "u",
            "v",
            "link_id",
            "length",
            "capacity",
            "free_flow_time_min",
            "speed_kph",
            "highway",
            "flow",
            "travel_time",
        ]
    ].copy()
    edge_export = edge_export.rename(
        columns={"free_flow_time_min": "free_flow_time", "highway": "road_type"}
    )
    edge_export.to_parquet(BASELINE_PARQUET, index=False)

    node_export = nodes[["node_id", "x", "y", "street_count"]].copy()
    node_export = node_export.rename(columns={"x": "lon", "y": "lat"})
    node_export.to_parquet(NODE_PARQUET, index=False)

    od_matrix.to_parquet(OD_PARQUET, index=True)


def visualize(edges: gpd.GeoDataFrame, output_path: Path = PLOT_PATH) -> None:
    """Create a static PNG highlighting congested flows."""

    edges = edges.copy()
    edges["flow"] = edges["flow"].fillna(0.0)
    max_flow = edges["flow"].max() or 1.0
    widths = 0.4 + 2.6 * (edges["flow"] / max_flow)

    fig, ax = plt.subplots(figsize=(8, 12))
    edges.plot(
        ax=ax,
        column="flow",
        linewidth=widths,
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Assigned flow (veh)"},
    )
    ax.set_axis_off()
    ax.set_title("Manhattan baseline UE assignment", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def detect_compute_device(console: Console) -> str:
    """Pick the best available compute device."""

    if torch is None:
        console.print("[yellow]PyTorch not installed; using CPU.[/yellow]")
        return "cpu"

    if torch.backends.mps.is_available():
        console.print("[green]Using Apple Metal Performance Shaders (MPS) for Torch ops.[/green]")
        return "mps"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        console.print(f"[green]Using CUDA GPU ({name}).[/green]")
        return "cuda"

    console.print("[yellow]No GPU backend available; using CPU.[/yellow]")
    return "cpu"


def summarize(console: Console, total_travel_time: float, edges: gpd.GeoDataFrame) -> None:
    """Print quick stats for sanity checking."""

    non_zero = edges[edges["flow"] > 0]
    table = Table(title="Assignment summary", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total travel time (veh-min)", f"{total_travel_time:,.0f}")
    table.add_row("Edges with flow", f"{non_zero.shape[0]:,d}")
    table.add_row("Max edge flow (veh)", f"{non_zero['flow'].max():,.0f}")
    table.add_row("Max travel time (min)", f"{non_zero['travel_time'].max():.2f}")
    console.print(table)


def main() -> None:
    set_global_state()
    console = Console()
    console.print("[bold]Starting Manhattan baseline simulation[/bold]")

    global COMPUTE_DEVICE
    COMPUTE_DEVICE = detect_compute_device(console)

    with console.status("Downloading and preparing network…"):
        G, nodes_gdf, edges_gdf = build_network(MANHATTAN_BBOX)
    console.print(
        f"[green]✓[/green] Downloaded Manhattan network "
        f"(nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,})"
    )

    with console.status("Generating synthetic OD demand…"):
        od_matrix = generate_od_matrix(G)
    console.print(f"[green]✓[/green] Generated OD matrix with {OD_PAIRS} OD pairs")

    with console.status("Running AequilibraE assignment…"):
        results = run_assignment(edges_gdf, od_matrix, console=console)
    console.print(
        f"[green]✓[/green] Total system travel time: "
        f"{results.total_travel_time:,.0f} vehicle-minutes"
    )

    save_results(results.edges, nodes_gdf, od_matrix)
    console.print(f"[green]✓[/green] Results saved to {BASELINE_PARQUET}")
    console.print(f"[green]✓[/green] Node features saved to {NODE_PARQUET}")
    console.print(f"[green]✓[/green] OD matrix saved to {OD_PARQUET}")

    visualize(results.edges, PLOT_PATH)
    console.print(f"[green]✓[/green] Visualization saved to {PLOT_PATH}")

    summarize(console, results.total_travel_time, results.edges)


if __name__ == "__main__":
    main()
