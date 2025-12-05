#!/bin/bash
# Example: Visualize traffic predictions from trained model

echo "=== Traffic Visualization Examples ==="
echo ""

echo "1. Visualizing latest model..."
python visualizations/traffic_heatmap.py \
    --model models/gnn_trained.pth \
    --output visualizations/current_traffic.png

echo ""
echo "2. Comparing all epochs..."
python visualizations/traffic_heatmap.py \
    --model models/gnn_trained.pth \
    --compare-epochs

echo ""
echo "3. Creating animated evolution (requires Pillow)..."
python visualizations/animation.py \
    --model-dir models \
    --output visualizations/training_evolution.gif

echo ""
echo "4. Generating PDF report..."
python visualizations/generate_report.py \
    --model-dir models \
    --output visualizations/training_report.pdf

echo ""
echo "Done! Check visualizations/ directory for outputs."
