"""Convert a local OSM .pbf file into a PyTorch Geometric graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import osmnx as ox
import networkx as nx
from pyrosm import OSM
import torch
from torch_geometric.data import Data

DEFAULT_SPEED_KPH = 35.0
SPEED_LOOKUP = {
    "motorway": 90.0,
    "trunk": 80.0,
    "primary": 60.0,
    "secondary": 45.0,
    "tertiary": 40.0,
    "residential": 30.0,
    "living_street": 25.0,
    "service": 20.0,
}
CAPACITY_LOOKUP = {
    "motorway": 4200,
    "trunk": 3600,
    "primary": 3200,
    "secondary": 2600,
    "tertiary": 2000,
    "residential": 900,
    "living_street": 600,
    "service": 400,
}


def normalize_highway(value) -> str:
    if isinstance(value, (list, tuple, set)):
        return next(iter(value))
    return value or "unclassified"


def load_graph_from_pbf(pbf_path: Path):
    """Load drivable network from a .pbf file via Pyrosm -> NetworkX."""

    print(f"Loading graph from {pbf_path} …")
    osm = OSM(str(pbf_path))
    nodes, edges = osm.get_network(network_type="driving", nodes=True, extra_attributes=["maxspeed", "oneway"])
    G = osm.to_graph(
        nodes,
        edges,
        graph_type="networkx",
        retain_all=False,
        osmnx_compatible=True,
    )
    if nx.is_directed(G):
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    else:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    G = ox.add_edge_speeds(G)
    return ox.graph_to_gdfs(G, nodes=True, edges=True)


def build_data_object(nodes_gdf, edges_gdf) -> Data:
    nodes = nodes_gdf.reset_index(drop=True)
    if "osmid" in nodes.columns:
        nodes = nodes.rename(columns={"osmid": "node_id"})
    else:
        nodes = nodes.rename(columns={"index": "node_id"})
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes["node_id"])}

    edges = edges_gdf.reset_index(drop=False).copy()
    edges["highway"] = edges["highway"].apply(normalize_highway)
    edges["speed_kph"] = edges["speed_kph"].fillna(edges["highway"].map(SPEED_LOOKUP)).fillna(DEFAULT_SPEED_KPH)
    edges["capacity"] = edges["highway"].map(CAPACITY_LOOKUP).fillna(1200)
    meters_per_minute = edges["speed_kph"] * (1000.0 / 60.0)
    edges["free_flow_time_min"] = edges["length"] / meters_per_minute.clip(lower=1.0)

    degree_counts = edges["u"].value_counts().add(edges["v"].value_counts(), fill_value=0)
    nodes["street_count"] = nodes["node_id"].map(degree_counts).fillna(0)

    node_feats = np.column_stack(
        [
            nodes["x"].to_numpy(float),
            nodes["y"].to_numpy(float),
            nodes["street_count"].fillna(0).to_numpy(float),
        ]
    ).astype(np.float32)

    edge_attr = np.column_stack(
        [
            edges["length"].to_numpy(float),
            edges["speed_kph"].to_numpy(float),
            edges["capacity"].to_numpy(float),
            edges["free_flow_time_min"].to_numpy(float),
        ]
    ).astype(np.float32)

    edge_index = np.vstack(
        [
            edges["u"].map(node_id_to_idx).to_numpy(int),
            edges["v"].map(node_id_to_idx).to_numpy(int),
        ]
    )

    data = Data(
        x=torch.from_numpy(node_feats),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr),
    )
    data.num_nodes = node_feats.shape[0]
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OSM .pbf to PyG graph")
    parser.add_argument("--pbf-path", type=str, required=True, help="Path to the .pbf file (e.g., data/nyc.osm.pbf)")
    parser.add_argument("--output", type=str, default="data/nyc_graph.pt", help="Destination .pt path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbf_path = Path(args.pbf_path)
    if not pbf_path.exists():
        raise SystemExit(f"PBF file not found: {pbf_path}")

    nodes_gdf, edges_gdf = load_graph_from_pbf(pbf_path)
    data = build_data_object(nodes_gdf, edges_gdf)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved PyG graph with {data.num_nodes:,} nodes and {data.edge_index.size(1):,} edges -> {output_path}")


if __name__ == "__main__":
    main()
