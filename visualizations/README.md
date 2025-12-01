# Traffic Visualization Module

Visualize GNN-predicted traffic patterns on a Manhattan street map.

## Usage

### 1. Visualize a Single Epoch

```bash
python visualizations/traffic_heatmap.py --model models/gnn_trained_epoch_5.pth
```

This creates `visualizations/traffic_map.png` showing:
- **Red edges**: High predicted travel time (congested)
- **Yellow edges**: Medium travel time
- **Green edges**: Low travel time (free-flowing)

### 2. Compare Multiple Epochs

```bash
python visualizations/traffic_heatmap.py --model models/gnn_trained.pth --compare-epochs
```

This generates separate visualizations for each epoch checkpoint found in `models/`:
- `visualizations/epoch_comparison/gnn_trained_epoch_1.png`
- `visualizations/epoch_comparison/gnn_trained_epoch_2.png`
- etc.

You can then view them side-by-side to see how the model's understanding of traffic evolved during training.

### 3. Generate PDF Report

```bash
python visualizations/generate_report.py --model-dir models
```

Creates `visualizations/training_report.pdf` containing:
- Title page
- One page per epoch showing the traffic heatmap
- Useful for sharing results or reviewing training progress offline.

### 4. Custom Output Path

```bash
python visualizations/traffic_heatmap.py \
    --model models/gnn_trained_optimized.pth \
    --output results/my_traffic_map.png
```

## How It Works

1. **Loads Model**: Loads the GNN from the specified checkpoint
2. **Predicts All Edges**: Runs inference to get predicted travel time for every road segment
3. **Maps to Colors**: Assigns colors based on predicted time (red=slow, green=fast)
4. **Plots on Map**: Draws each edge on a coordinate plot with its corresponding color

## Interpretation

- **Dark Red Clusters**: Areas where the model predicts high congestion
- **Green Networks**: Routes the model thinks are fast
- **Evolution Across Epochs**: Early epochs may show uniform predictions; later epochs should show realistic traffic patterns

## Requirements

Install visualization dependencies:
```bash
pip install matplotlib
```
