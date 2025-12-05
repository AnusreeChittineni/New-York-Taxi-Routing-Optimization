"""
Create an animated GIF showing how traffic predictions evolve across training epochs.
"""

import sys
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gnn.gnn_model import load_model, detect_device


def create_epoch_animation(
    graph_path: str,
    model_dir: str = "models",
    output_path: str = "visualizations/training_evolution.gif",
    duration_ms: int = 500,
):
    """
    Create an animated GIF showing traffic prediction evolution across epochs.
    
    Args:
        graph_path: Path to graph file
        model_dir: Directory containing epoch checkpoints
        output_path: Where to save the GIF
        duration_ms: Milliseconds per frame
    """
    # Find all epoch checkpoints
    pattern = str(Path(model_dir) / "*_epoch_*.pth")
    model_paths = sorted(glob.glob(pattern))
    
    if not model_paths:
        print(f"No epoch checkpoints found in {model_dir}")
        return
    
    print(f"Found {len(model_paths)} epoch checkpoints")
    
    # Load graph once
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    device = detect_device()
    
    in_channels = data.num_node_features
    edge_attr_dim = data.edge_attr.size(1) if hasattr(data, 'edge_attr') else 0
    edge_index = data.edge_index.cpu().numpy()
    node_coords = data.x[:, :2].cpu().numpy()
    
    data = data.to(device)
    
    # Generate frames
    frames = []
    for i, model_path in enumerate(model_paths):
        print(f"Processing epoch {i+1}/{len(model_paths)}...")
        
        # Load model
        model = load_model(model_path, in_channels, hidden_channels=64, edge_attr_dim=edge_attr_dim)
        model.to(device)
        model.eval()
        
        # Predict
        with torch.no_grad():
            edge_pred = model(data).detach().cpu().numpy()
        
        # Create frame
        fig, ax = plt.subplots(figsize=(10, 14))
        
        norm = Normalize(vmin=0, vmax=60)
        colormap = cm.get_cmap('RdYlGn_r')
        
        # Plot edges (sample for speed)
        sample_indices = np.random.choice(edge_index.shape[1], size=min(50000, edge_index.shape[1]), replace=False)
        
        for idx in sample_indices:
            src_idx = edge_index[0, idx]
            dst_idx = edge_index[1, idx]
            
            src_coords = node_coords[src_idx]
            dst_coords = node_coords[dst_idx]
            
            time = edge_pred[idx]
            color = colormap(norm(time))
            
            ax.plot(
                [src_coords[0], dst_coords[0]],
                [src_coords[1], dst_coords[1]],
                color=color,
                alpha=0.5,
                linewidth=0.4,
            )
        
        # Format
        epoch_name = Path(model_path).stem.split('_')[-1]  # Extract epoch number
        ax.set_title(f'Training Epoch {epoch_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        # Save frame to buffer
        temp_path = f"/tmp/frame_{i:03d}.png"
        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(Image.open(temp_path))
    
    # Create GIF
    print(f"Creating animation with {len(frames)} frames...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training evolution animation")
    parser.add_argument("--graph", type=str, default="data/nyc_graph.pt")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output", type=str, default="visualizations/training_evolution.gif")
    parser.add_argument("--duration", type=int, default=500, help="Milliseconds per frame")
    
    args = parser.parse_args()
    
    create_epoch_animation(args.graph, args.model_dir, args.output, args.duration)
