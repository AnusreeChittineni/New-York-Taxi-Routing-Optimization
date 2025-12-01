#!/bin/bash
# Optimized training script with better hyperparameters

echo "Starting optimized GNN training..."
echo "Changes:"
echo "  - Policy: Greedy (simpler, more stable)"
echo "  - Learning Rate: 1e-4 (lower, prevents overshooting)"
echo "  - Eval Samples: 2000 (more stable metrics)"
echo ""

python gnn/train_gnn.py \
    --policy-mode complex \
    --POLICY greedy \
    --lr 1e-4 \
    --rmse-eval-samples 2000 \
    --epochs 10 \
    --train-sample-count 50000 \
    --val-sample-count 1000 \
    --batch-size 8 \
    --model-path models/gnn_trained_optimized.pth \
    --stats-path training_stats_optimized.csv
