"""
Visualization utilities for GNN-predicted traffic patterns.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib.colors import Normalize

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gnn.gnn_model import load_model, detect_device


def visualize_traffic_heatmap(
    graph_path: str,
    model_path: str,
    output_path: str | None = "visualizations/traffic_map.png",
    figsize: tuple = (12, 16),
    cmap: str = "RdYlGn_r",  # Red (high) to Green (low)
):
    """
    Visualize GNN-predicted travel times as a traffic heatmap on Manhattan streets.
    
    Args:
        graph_path: Path to the graph file (e.g., data/nyc_graph.pt)
        model_path: Path to model checkpoint (e.g., models/gnn_trained_epoch_5.pth)
        output_path: Where to save the visualization
        figsize: Figure size (width, height)
        cmap: Colormap for travel times (default: Red=slow, Green=fast)
    """
    # Load graph
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    
    # Load model
    print(f"Loading model from {model_path}...")
    device = detect_device()
    in_channels = data.num_node_features
    edge_attr_dim = data.edge_attr.size(1) if hasattr(data, 'edge_attr') else 0
    
    model = load_model(model_path, in_channels, hidden_channels=64, edge_attr_dim=edge_attr_dim)
    model.to(device)
    model.eval()
    
    # Predict travel times
    print("Predicting edge travel times...")
    data = data.to(device)
    with torch.no_grad():
        edge_pred = model(data).detach().cpu().numpy()
    
    # Extract edge coordinates
    edge_index = data.edge_index.cpu().numpy()
    node_coords = data.x[:, :2].cpu().numpy()  # Assuming first 2 features are (x, y)
    
    # Compute statistics
    min_time = edge_pred.min()
    max_time = edge_pred.max()
    mean_time = edge_pred.mean()
    
    print(f"Predicted times - Min: {min_time:.2f}, Max: {max_time:.2f}, Mean: {mean_time:.2f} min")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize colors
    norm = Normalize(vmin=0, vmax=min(60, max_time))  # Cap at 60 min for better contrast
    colormap = cm.get_cmap(cmap)
    
    # Plot edges
    print(f"Plotting {edge_index.shape[1]} edges...")
    for i in range(edge_index.shape[1]):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{edge_index.shape[1]}")
        
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        
        src_coords = node_coords[src_idx]
        dst_coords = node_coords[dst_idx]
        
        time = edge_pred[i]
        color = colormap(norm(time))
        
        # Draw edge with color based on predicted time
        ax.plot(
            [src_coords[0], dst_coords[0]],
            [src_coords[1], dst_coords[1]],
            color=color,
            alpha=0.6,
            linewidth=0.5,
        )
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Predicted Travel Time (min)')
    
    # Format plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'GNN Traffic Prediction\nModel: {Path(model_path).name}')
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Save or return
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()
        return None
    else:
        return fig


def compare_epochs(
    graph_path: str,
    model_paths: list[str],
    output_dir: str = "visualizations/epoch_comparison",
):
    """
    Generate side-by-side comparison of traffic predictions across epochs.
    
    Args:
        graph_path: Path to graph file
        model_paths: List of model checkpoint paths (e.g., different epochs)
        output_dir: Directory to save comparison images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate individual visualizations
    for model_path in model_paths:
        epoch_name = Path(model_path).stem  # e.g., "gnn_trained_epoch_5"
        output_path = f"{output_dir}/{epoch_name}.png"
        visualize_traffic_heatmap(graph_path, model_path, output_path)
    
    print(f"\nGenerated {len(model_paths)} epoch visualizations in {output_dir}/")
    print("You can compare them side-by-side to see how predictions evolved.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize GNN traffic predictions")
    parser.add_argument("--graph", type=str, default="data/nyc_graph.pt", help="Path to graph file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="visualizations/traffic_map.png", help="Output image path")
    parser.add_argument("--compare-epochs", action="store_true", help="Compare multiple epoch checkpoints")
    
    args = parser.parse_args()
    
    if args.compare_epochs:
        # Auto-detect all epoch checkpoints
        import glob
        model_dir = Path(args.model).parent
        pattern = str(model_dir / "*_epoch_*.pth")
        models = sorted(glob.glob(pattern))
        
        if not models:
            print(f"No epoch checkpoints found matching {pattern}")
            sys.exit(1)
        
        print(f"Found {len(models)} epoch checkpoints")
        compare_epochs(args.graph, models)
    else:
        visualize_traffic_heatmap(args.graph, args.model, args.output)
