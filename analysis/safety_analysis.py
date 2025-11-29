"""
Analyze the impact of avoiding dangerous road segments on travel times.
"""

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gnn.gnn_model import load_model, detect_device
from gnn.routing import build_nx_from_pyg, k_shortest_paths, path_edge_mask

def load_safety_data(csv_path: str) -> set[int]:
    """
    Load safety data and return a set of OSM IDs to avoid.
    We avoid 'Critical Bottleneck' and 'Hidden Danger' segments.
    """
    df = pd.read_csv(csv_path)
    # Filter for All_Day and dangerous categories
    dangerous_categories = ["Critical Bottleneck", "Hidden Danger"]
    mask = (df["Time_Period"] == "All_Day") & (df["Category"].isin(dangerous_categories))
    bad_segments = df[mask]["SegmentID"].unique()
    return set(bad_segments)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", type=str, default="data/nyc_graph.pt")
    parser.add_argument("--model-path", type=str, default="models/gnn_trained.pth")
    parser.add_argument("--safety-csv", type=str, default="data/nyc_road_safety_analysis.csv")
    parser.add_argument("--num-trips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = detect_device()
    print(f"Using device: {device}")

    # Load Graph
    print(f"Loading graph from {args.graph_path}...")
    data = torch.load(args.graph_path, weights_only=False)
    
    if not hasattr(data, "edge_osm_id"):
        raise ValueError("Graph does not have 'edge_osm_id' attribute. Please rebuild graph.")

    # The trained model expects 2 node features (x, y), but the new graph has 3 (x, y, street_count).
    # We slice the features to match the model.
    if data.num_node_features == 3:
        print("Slicing node features to match model input (3 -> 2)...")
        data.x = data.x[:, :2]

    # Load Model
    print(f"Loading model from {args.model_path}...")
    # We need to know hidden channels. Assuming 64 from default training args.
    # Ideally we should save config with model.
    model = load_model(args.model_path, data.num_node_features, hidden_channels=64, edge_attr_dim=data.edge_attr.size(1))
    model.to(device)
    model.eval()

    # Predict Edge Times
    print("Predicting edge travel times...")
    data = data.to(device)
    with torch.no_grad():
        edge_pred = model(data).detach().cpu().numpy()
    
    # Build NetworkX Graph
    print("Building NetworkX graph...")
    # We need to pass edge_osm_id to the graph
    # build_nx_from_pyg doesn't natively support custom attributes easily unless we modify it
    # or we can just add it after.
    # Let's modify build_nx_from_pyg or just iterate.
    # Actually, let's just use the edge_index and our predicted weights.
    
    G = nx.DiGraph()
    edge_index = data.edge_index.cpu().numpy()
    edge_osm_ids = data.edge_osm_id.cpu().numpy()
    
    num_edges = edge_index.shape[1]
    for i in tqdm(range(num_edges), desc="Building Graph"):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        osm_id = int(edge_osm_ids[i])
        weight = float(edge_pred[i])
        # We store edge_id to map back if needed, but here we just need weight and osm_id
        G.add_edge(u, v, weight=weight, osm_id=osm_id)

    # Identify Dangerous Edges
    print(f"Loading safety data from {args.safety_csv}...")
    bad_osm_ids = load_safety_data(args.safety_csv)
    print(f"Found {len(bad_osm_ids)} dangerous segments.")

    # Count how many edges in graph match bad segments
    graph_bad_edges = []
    for u, v, d in G.edges(data=True):
        if d.get("osm_id") in bad_osm_ids:
            graph_bad_edges.append((u, v))
    
    print(f"Graph contains {len(graph_bad_edges)} edges marked as dangerous.")

    if not graph_bad_edges:
        print("Warning: No dangerous edges found in the graph. Check ID mapping.")
        # Proceed anyway to show 0 diff

    # Create Safe Graph (Copy and remove edges)
    G_safe = G.copy()
    G_safe.remove_edges_from(graph_bad_edges)

    # Sample Trips
    print(f"Sampling {args.num_trips} random trips...")
    nodes = list(G.nodes())
    trips = []
    for _ in range(args.num_trips):
        o, d = random.sample(nodes, 2)
        trips.append((o, d))

    # Compare Travel Times
    print("Comparing travel times...")
    times_full = []
    times_safe = []
    
    success_count = 0
    
    for o, d in tqdm(trips):
        try:
            # Shortest path on full graph
            len_full = nx.shortest_path_length(G, o, d, weight="weight")
            
            # Shortest path on safe graph
            try:
                len_safe = nx.shortest_path_length(G_safe, o, d, weight="weight")
            except nx.NetworkXNoPath:
                # If no path in safe graph, we can't compare directly. 
                # Maybe skip or assign penalty? Let's skip for average.
                continue
                
            times_full.append(len_full)
            times_safe.append(len_safe)
            success_count += 1
        except nx.NetworkXNoPath:
            continue

    if success_count == 0:
        print("No valid paths found.")
        return

    avg_full = np.mean(times_full)
    avg_safe = np.mean(times_safe)
    diff = avg_safe - avg_full
    pct_diff = (diff / avg_full) * 100 if avg_full > 0 else 0

    print("\nResults:")
    print(f"Evaluated {success_count} trips.")
    print(f"Average Travel Time (Full Graph): {avg_full:.2f} min")
    print(f"Average Travel Time (Safe Graph): {avg_safe:.2f} min")
    print(f"Difference: {diff:+.2f} min ({pct_diff:+.2f}%)")
    
    if diff > 0:
        print("Avoiding dangerous roads INCREASED travel time.")
    else:
        print("Avoiding dangerous roads DECREASED travel time (unexpected, unless congestion was high on them).")

if __name__ == "__main__":
    main()
