"""
Identify and list the most congested ('bad') edges in the city based on GNN predictions.
This script is standalone and does not require modifying the training code.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gnn.gnn_model import load_model, detect_device


def find_bad_edges(
    graph_path: str,
    model_path: str,
    output_path: str = "results/bad_edges.csv",
    top_k: int = 50,
):
    """
    Predict travel times for all edges and save the slowest ones to a CSV.
    
    Args:
        graph_path: Path to graph file
        model_path: Path to trained model
        output_path: Path to save the CSV report
        top_k: Number of top congested edges to list
    """
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    
    print(f"Loading model from {model_path}...")
    device = detect_device()
    in_channels = data.num_node_features
    edge_attr_dim = data.edge_attr.size(1) if hasattr(data, 'edge_attr') else 0
    
    model = load_model(model_path, in_channels, hidden_channels=64, edge_attr_dim=edge_attr_dim)
    model.to(device)
    model.eval()
    
    print("Predicting travel times...")
    data = data.to(device)
    with torch.no_grad():
        # Get predicted travel times (in minutes)
        pred_times = model(data).detach().cpu().numpy()
    
    # Create a DataFrame
    edges_df = pd.DataFrame({
        'edge_index': range(len(pred_times)),
        'predicted_time_min': pred_times,
        'u': data.edge_index[0].cpu().numpy(),
        'v': data.edge_index[1].cpu().numpy(),
    })
    
    # Add coordinates if available
    if hasattr(data, 'x'):
        coords = data.x[:, :2].cpu().numpy() # (lon, lat) usually
        edges_df['u_lon'] = coords[edges_df['u'], 0]
        edges_df['u_lat'] = coords[edges_df['u'], 1]
        edges_df['v_lon'] = coords[edges_df['v'], 0]
        edges_df['v_lat'] = coords[edges_df['v'], 1]
        
    # Sort by predicted time (descending) to find "bad" edges
    # We look for edges with high travel times relative to their length if possible,
    # but raw high travel time is also a good indicator of bottlenecks.
    
    if hasattr(data, 'edge_attr') and data.edge_attr.size(1) > 0:
        # Assuming first attribute is distance
        distances = data.edge_attr[:, 0].cpu().numpy()
        edges_df['distance'] = distances
        
        # Calculate "slowness" (minutes per unit distance)
        # Avoid div by zero
        edges_df['slowness'] = edges_df['predicted_time_min'] / (edges_df['distance'] + 1e-6)
        
        # Sort by slowness (most congested)
        print("Sorting by congestion (time per unit distance)...")
        bad_edges = edges_df.sort_values('slowness', ascending=False).head(top_k)
    else:
        print("Sorting by raw travel time (longest duration)...")
        bad_edges = edges_df.sort_values('predicted_time_min', ascending=False).head(top_k)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bad_edges.to_csv(output_path, index=False)
    print(f"\nTop {top_k} bad edges saved to {output_path}")
    print("\nSample of bad edges:")
    print(bad_edges[['u', 'v', 'predicted_time_min']].head())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="data/nyc_graph.pt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/bad_edges.csv")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    
    find_bad_edges(args.graph, args.model, args.output, args.top_k)
