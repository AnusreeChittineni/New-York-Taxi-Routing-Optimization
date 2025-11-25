# ⚡️ NYC Traffic GNN - Quick Reference

## 🚀 Quick Start
```bash
# Train model
python train_gnn.py --K_CANDS 8 --POLICY gumbel_softmax --LOSS huber

# Evaluate closures (Beam Search)
python test_gnn_beam.py --beam_width 5 --steps 10

# Visualize results
python -m matplotlib results/plots/beam_search.png
```

## 📂 Key Files
| File | Purpose |
|------|----------|
| `gnn_model.py` | GNN architecture (GraphSAGE/GAT) & regression head |
| `train_gnn.py` | Training loop with stochastic candidate route policy |
| `exploration_policy.py` | Policies: ε-greedy, Softmax, Gumbel-Softmax |
| `routing.py` | Yen's K-shortest paths & path masking |
| `test_gnn_beam.py` | Road closure evaluation via Beam Search |

## 🧠 Model Architecture
- **Input**: Node features (coords, degree) & Edge features (length, capacity, flow)
- **Encoder**: GraphSAGE / GAT layers
- **Output**: Predicted edge travel time $\hat{t}(e)$
- **Objective**: Minimize difference between predicted path time and observed trip time ($T^{\text{obs}}$)

## 🔬 Training Parameters
| Param | Description | Typical Range |
|-------|-------------|---------------|
| `K_CANDS` | Number of candidate paths per OD pair | 5 - 10 |
| `POLICY` | Exploration policy | `epsilon_greedy`, `softmax`, `gumbel_softmax` |
| `LOSS` | Loss function | `mse`, `huber` |
| `epsilon` | Random exploration rate | 0.3 → 0.02 |
| `tau` | Temperature (Softmax/Gumbel) | 2.0 → 0.25 |

## 📊 Evaluation Metrics
- **AvgTT**: Average Travel Time across all trips (using best predicted path)
- **WAvgTT**: Weighted Average Travel Time (weighted by hotspot demand)
