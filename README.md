# 🚦 NYC Graph Neural Network Traffic Simulation & Optimization

## Overview

This project builds an intelligent **traffic simulation and optimization framework** for Manhattan using:
- **AequilibraE** for macroscopic equilibrium simulation
- **DuckDB** + TLC Yellow Taxi data for real trip observations
- **PyTorch Geometric** for graph-based learning
- A **stochastic candidate route policy** for exploration-based training

The system learns to **predict edge-level travel times** and **infer plausible routes** without ground-truth path labels. It also allows **testing of road closures** through beam search and computing travel-time metrics.

---

## 🚀 End-to-End Workflow

### 1️⃣ Data and Graph Creation
- Manhattan road network downloaded via **OSMnx**
- Converted to a **PyTorch Geometric** graph (`manhattan_graph.pt`)
- Edge features: length, capacity, free-flow time, flow
- Node features: coordinates, degree, centrality metrics

### 2️⃣ Data Source
Real trip data comes from **DuckDB** tables (NYC TLC Yellow Taxi data):
- `PULocationID`, `DOLocationID` → origin/destination zones
- `tpep_pickup_datetime`, `tpep_dropoff_datetime` → observed trip durations
- `trip_distance`, `total_amount` → auxiliary features

### 3️⃣ GNN Model
- **Input**: Node and edge features
- **Output**: Predicted edge-level travel times $\hat{t}(e)$
- **Architecture**: GraphSAGE / GAT layers, ReLU activations

### 4️⃣ Candidate Route Generation
- For each OD pair, generate **K candidate paths** using **Yen's K-shortest paths** algorithm
- Edge weights come from current model predictions
- Encourage diversity via overlap penalty:

$$w^{(\text{div})}_e = \hat{t}(e) + \lambda \cdot \phi_e$$

where $\phi_e$ counts how many prior candidates included edge $e$.

### 5️⃣ Exploration-based Path Selection
Since true routes are unknown, the model maintains a **policy** over candidate paths.

We want the model to **explore early** (many routes) and **exploit later** (best route).

---

## 🧠 Policy Space and Training Math

Let:
- $\hat{t}(e)$ — predicted edge travel time
- $s_{ij} = \sum_{e \in p_{ij}} \hat{t}(e)$ — predicted time for path $p_{ij}$
- $T_i^{\text{obs}} = t_{\text{dropoff}} - t_{\text{pickup}}$ — observed trip duration

We define a **policy** $\pi_\theta(j | \mathcal{P}_i)$ over the candidate routes $\mathcal{P}_i = \{p_{i1}, \dots, p_{iK}\}$.

---

### 1. Candidate Path Generation

Using current edge costs, produce $K$ paths per OD pair:

$$\mathcal{P}_i = \{p_{i1}, p_{i2}, \ldots, p_{iK}\}$$

Each has a predicted cost $s_{ij}$.

---

### 2. Policy Definitions

#### (a) ε-Greedy (Exploration–Exploitation Tradeoff)

$$j \sim \begin{cases}
\arg\min_j s_{ij}, & \text{with probability } 1-\varepsilon \\
\text{Uniform}(1..K), & \text{with probability } \varepsilon
\end{cases}$$

- Anneal $\varepsilon_t$ from 0.3 → 0.02 across training
- Ensures exploration early, exploitation later

#### (b) Softmax (Boltzmann Policy)

$$\pi_\tau(j) = \frac{\exp(-s_{ij}/\tau)}{\sum_k \exp(-s_{ik}/\tau)}$$

- Temperature $\tau$ controls randomness
  - High $\tau$: almost uniform (explore)
  - Low $\tau$: near-greedy (exploit)

Expected path time:

$$\tilde{T}_i = \sum_j \pi_\tau(j)\, s_{ij}$$

#### (c) Gumbel–Softmax (Differentiable Approximation)

Adds Gumbel noise $g_j \sim \text{Gumbel}(0,1)$:

$$\alpha_j = \frac{\exp\left( (-s_{ij} + g_j)/\tau \right)}{\sum_k \exp\left( (-s_{ik} + g_k)/\tau \right)}$$

Soft mask over edges:

$$m^{(\text{soft})}(e) = \sum_j \alpha_j \cdot \mathbf{1}\{e \in p_{ij}\}$$

Soft path time:

$$\tilde{T}_i = \sum_{e} m^{(\text{soft})}(e) \hat{t}(e) = \sum_j \alpha_j s_{ij}$$

Differentiable and trainable end-to-end.

---

### 3. Loss Function

For trip $i$:
- **Hard selection**:

$$\mathcal{L}_i = (s_{ij} - T_i^{\text{obs}})^2$$

- **Soft mixture**:

$$\mathcal{L}_i = (\tilde{T}_i - T_i^{\text{obs}})^2$$

Optionally use **Huber loss** for robustness:

$$\ell_\delta(a,b) = \begin{cases}
\frac{1}{2}(a-b)^2, & |a-b| \le \delta \\
\delta(|a-b| - \tfrac{1}{2}\delta), & \text{otherwise}
\end{cases}$$

#### + Entropy Bonus

Keeps exploration active:

$$\mathcal{L}_\text{entropy} = -\beta H(\pi), \quad H(\pi) = -\sum_j \pi(j)\log\pi(j)$$

Final loss:

$$\mathcal{L} = \frac{1}{N}\sum_i [\mathcal{L}_i + \mathcal{L}_\text{entropy}]$$

---

### 4. Gradient Flow

- **Hard policies** (ε-greedy): gradients flow through selected edges only
- **Soft policies** (softmax, Gumbel): gradients flow through mixture weights

Gumbel–Softmax allows near-REINFORCE exploration without high variance.

---

### 5. Exploration Scheduling

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| ε (epsilon) | Random exploration rate | 0.3 → 0.02 |
| τ (tau) | Temperature for softmax / Gumbel | 2.0 → 0.25 |
| β (entropy) | Weight for entropy bonus | 0.001–0.01 |

Schedules:

$$\varepsilon_t = \varepsilon_\text{end} + (\varepsilon_\text{start}-\varepsilon_\text{end})\left(1 - \frac{t}{T}\right)$$

$$\tau_t = \tau_\text{end} + (\tau_\text{start}-\tau_\text{end})\left(1 - \frac{t}{T}\right)$$

---

### 6. Beam Search for Edge Removal

At test time, the system simulates **road closures**.

#### Beam Search Algorithm:
1. Maintain a beam of top-B graph configurations (edges removed)
2. At each step:
   - Remove one new edge from each configuration
   - Evaluate metrics (below)
   - Keep top-B by weighted travel time
3. Continue until K edges removed

#### Metrics:

**1. Average Travel Time (AvgTT):**

$$\text{AvgTT} = \frac{1}{N}\sum_i \min_j \sum_{e\in p_{ij}} \hat{t}(e)$$

**2. Weighted Average Travel Time (WAvgTT):**

$$\text{WAvgTT} = \mathbb{E}_{(o,d)\sim q_\text{hotspot}}\left[\min_j \sum_{e\in p_{ij}}\hat{t}(e)\right]$$

where $q_\text{hotspot}$ is built from pickup/dropoff frequencies in DuckDB.

---

## ⚙️ Training Loop Summary

1. Load OD pairs & observed trip durations from DuckDB
2. For each mini-batch:
   - Generate K candidate paths per OD
   - Compute predicted times $s_{ij}$
   - Sample route $j$ from the exploration policy
   - Compute loss $(\hat{T} - T^{\text{obs}})^2$
   - Backpropagate into GNN edge-time predictions
3. Gradually reduce exploration (ε or τ)
4. Save model weights

---

## 🔬 Mathematical Summary

| Symbol | Meaning |
|--------|---------|
| $\hat{t}(e)$ | Predicted edge travel time |
| $s_{ij} = \sum_{e\in p_{ij}} \hat{t}(e)$ | Predicted path travel time |
| $T_i^{\text{obs}}$ | Observed travel time (dropoff–pickup) |
| $\pi(j)$ | Policy probability of choosing path j |
| $\alpha_j$ | Gumbel-Softmax mixture coefficient |
| $m(e)$ | Path mask (1 if edge e on path) |
| $\tilde{T}_i$ | Expected path time |
| $\mathcal{L}_i$ | Trip-level loss |
| $\varepsilon, \tau, \beta$ | Exploration parameters |

---

## 🧭 Conceptual Intuition

- **Early epochs**: The policy tries many plausible routes → exploration
- **Mid training**: Model begins predicting more accurate edge times → exploitation
- **Late training**: Routes stabilize, representing realistic driver behavior under traffic

This process approximates **inverse reinforcement learning**: we don't observe the policy (route choices), only outcomes (trip durations). The model **infers latent routes** consistent with observed trip times.

---

## 📊 Evaluation

Use `test_gnn_beam.py` to evaluate road-closure impacts.

Example:
```bash
python test_gnn_beam.py --beam_width 5 --steps 10
```

---

## 🚀 Quick Start

```bash
# Step 1 — Train model
python train_gnn.py --K_CANDS 8 --POLICY gumbel_softmax --LOSS huber

# Step 2 — Evaluate closures
python test_gnn_beam.py --beam_width 5 --steps 10

# Step 3 — Visualize
python -m matplotlib results/plots/beam_search.png
```
