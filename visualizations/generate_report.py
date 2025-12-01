"""
Generate a PDF report compiling traffic visualizations across all training epochs.
"""

import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from visualizations.traffic_heatmap import visualize_traffic_heatmap


def generate_pdf_report(
    graph_path: str,
    model_dir: str = "models",
    output_path: str = "visualizations/training_report.pdf",
):
    """
    Generate a multi-page PDF report with traffic visualizations for each epoch.
    
    Args:
        graph_path: Path to graph file
        model_dir: Directory containing epoch checkpoints
        output_path: Where to save the PDF report
    """
    # Find all epoch checkpoints
    pattern = str(Path(model_dir) / "*_epoch_*.pth")
    model_paths = sorted(glob.glob(pattern))
    
    if not model_paths:
        print(f"No epoch checkpoints found in {model_dir}")
        return
    
    print(f"Found {len(model_paths)} epoch checkpoints. Generating report...")
    
    # Create PDF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Title Page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.6, "GNN Traffic Prediction Training Report", 
                 ha='center', va='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.4, f"Generated from {len(model_paths)} epochs", 
                 ha='center', va='center', fontsize=16)
        pdf.savefig()
        plt.close()
        
        # Generate page for each epoch
        for i, model_path in enumerate(model_paths):
            print(f"Processing epoch {i+1}/{len(model_paths)}: {Path(model_path).name}...")
            
            # Get figure from visualization function (pass output_path=None)
            fig = visualize_traffic_heatmap(
                graph_path=graph_path,
                model_path=model_path,
                output_path=None,
                figsize=(10, 12)  # Slightly smaller to fit on page
            )
            
            if fig:
                # Add epoch title if not already clear
                epoch_name = Path(model_path).stem
                fig.suptitle(f"Epoch: {epoch_name}", fontsize=16, y=0.95)
                
                # Save to PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
    print(f"Successfully generated PDF report at: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PDF report of training evolution")
    parser.add_argument("--graph", type=str, default="data/nyc_graph.pt")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output", type=str, default="visualizations/training_report.pdf")
    
    args = parser.parse_args()
    
    generate_pdf_report(args.graph, args.model_dir, args.output)
