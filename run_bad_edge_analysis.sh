#!/bin/bash
# Run analysis to find bad edges (bottlenecks)

echo "Identifying top 50 bad edges..."
python analysis/find_bad_edges.py \
    --model models/gnn_trained_optimized.pth \
    --output results/bad_edges.csv \
    --top-k 50

echo ""
echo "Analysis complete. Results saved to results/bad_edges.csv"
