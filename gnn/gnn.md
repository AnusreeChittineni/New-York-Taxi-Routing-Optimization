# 🧠 Graph Neural Network (GNN) for Traffic Flow Prediction

## Overview

This document explains the GNN used in the NYC Traffic Simulation project — its architecture, learning objective, and training policy. The model predicts **edge-level travel times** from **node and edge features** of the Manhattan road network. It serves as the foundation for estimating travel times under normal conditions and evaluating **road closure impacts**.

---

## ⚙️ Core Design

The model uses a **graph-based representation** of Manhattan, where:

- **Nodes ($V$)** represent intersections
- **Edges ($E$)** represent road segments between intersections
- Each edge has associated features:
  - Length
  - Capacity
  - Free-flow time
  - Current flow
  - Historical average speed
- Each node contains attributes like coordinates, degree, and betweenness centrality

The GNN learns a mapping:

$$f_\theta: (V, E, X_v, X_e) \rightarrow \hat{t}(e)$$

that predicts the **travel time** $\hat{t}(e)$ for each edge.

---

## 🧩 Model Architecture

### 1. Input
- Node feature matrix: $X_v \in \mathbb{R}^{|V| \times d_v}$
- Edge feature matrix: $X_e \in \mathbb{R}^{|E| \times d_e}$
- Graph connectivity: adjacency lists from OSM (AequilibraE network)

### 2. Encoder
We use **GraphSAGE** (alternatively, GAT or GIN) layers to aggregate neighborhood features:

$$h_v^{(k+1)} = \sigma\left(W_1 h_v^{(k)} + W_2 \cdot \text{AGG}\big(\{h_u^{(k)} : u \in \mathcal{N}(v)\}\big)\right)$$

where:
- $\text{AGG}(\cdot)$ is a mean or attention-based aggregator
- $\sigma$ is ReLU activation
- $h_v^{(k)}$ represents the hidden representation of node $v$ at layer $k$

### 3. Edge Regression Head
For an edge $e = (u, v)$:

$$z_e = [h_u^{(K)} \, \| \, h_v^{(K)} \, \| \, X_e]$$

$$\hat{t}(e) = \text{MLP}(z_e)$$

where the MLP predicts the edge's travel time.

### 4. Output
Vector of predicted travel times for all edges:

$$\hat{\boldsymbol{t}} = [\hat{t}(e_1), \hat{t}(e_2), ..., \hat{t}(e_M)]$$

---

## 🚦 Routing Objective

The GNN is not trained on labeled paths — only **start/end** locations and **observed travel durations**. We simulate the latent route discovery process through **candidate path generation** and **policy-based exploration**.

---

## 🔍 Candidate Path Generation

For each trip:
- Origin–destination (OD) pair: $(o_i, d_i)$
- Generate $K$ candidate paths using **Yen's K-shortest paths**
- Compute predicted travel times:

$$s_{ij} = \sum_{e \in p_{ij}} \hat{t}(e)$$

where $p_{ij}$ is the $j^\text{th}$ candidate path for trip $i$.

The ground-truth travel time is:

$$T_i^{\text{obs}} = t_{\text{dropoff}} - t_{\text{pickup}}$$

---

## 🎯 Learning Objective (Policy-based Route Selection)

Because the true route is unknown, the model learns to select plausible routes using a **stochastic policy** over candidates.

### 1️⃣ Policy Distribution
Define the probability of choosing path $j$ for OD pair $i$:

$$\pi_\tau(j) = \frac{\exp(-s_{ij}/\tau)}{\sum_k \exp(-s_{ik}/\tau)}$$

- $\tau$: temperature (higher = more exploration)
- As $\tau \to 0$, selection becomes greedy (lowest predicted time)

---

### 2️⃣ Path Sampling Variants

#### ε-Greedy (Discrete)

$$j = \begin{cases}
\arg\min_j s_{ij}, & \text{with prob } 1-\varepsilon \\
\text{random}(1..K), & \text{with prob } \varepsilon
\end{cases}$$

- Used for stable, exploration–exploitation balance

#### Softmax / Boltzmann (Differentiable)

$$\tilde{T}_i = \sum_j \pi_\tau(j) \, s_{ij}$$

#### Gumbel–Softmax (Continuous Relaxation)

$$\alpha_j = \frac{\exp(( -s_{ij} + g_j)/\tau)}{\sum_k \exp(( -s_{ik} + g_k)/\tau)}, \quad g_j \sim \text{Gumbel}(0,1)$$

$$\tilde{T}_i = \sum_j \alpha_j \, s_{ij}$$

This allows **end-to-end differentiable training** through stochastic routing.

---

## 📘 Loss Function

The main loss aligns predicted and observed travel times.

$$\mathcal{L}_i = (\tilde{T}_i - T_i^{\text{obs}})^2$$

Optionally use **Huber loss** for robustness:

$$\ell_\delta(a,b) = \begin{cases}
\frac{1}{2}(a-b)^2, & |a-b| \le \delta \\
\delta(|a-b| - \frac{1}{2}\delta), & \text{otherwise}
\end{cases}$$

Add an **entropy bonus** to encourage exploration:

$$\mathcal{L}_\text{entropy} = -\beta H(\pi), \quad H(\pi) = -\sum_j \pi(j) \log \pi(j)$$

Full batch loss:

$$\mathcal{L} = \frac{1}{N}\sum_i [\mathcal{L}_i + \mathcal{L}_\text{entropy}]$$

---

## 📉 Optimization & Training

- **Optimizer**: `AdamW` with learning rate decay
- Gradients flow through selected paths (hard) or mixture weights (soft)
- Exploration parameters anneal over time:
  - $\varepsilon_t: 0.3 \rightarrow 0.02$
  - $\tau_t: 2.0 \rightarrow 0.25$

---

## 🧮 Evaluation Metrics

### 1. Average Travel Time (AvgTT)

$$\text{AvgTT} = \frac{1}{N}\sum_i \min_j \sum_{e\in p_{ij}} \hat{t}(e)$$

Measures network-wide efficiency.

### 2. Weighted Average Travel Time (WAvgTT)

$$\text{WAvgTT} = \mathbb{E}_{(o,d)\sim q_\text{hotspot}}\left[\min_j \sum_{e\in p_{ij}}\hat{t}(e)\right]$$

Weighted by demand distribution (hotspots from DuckDB trip frequencies).

---

## 🧠 Beam Search Evaluation (Road Closure Simulation)

During evaluation, the model is used to assess **impact of edge removals** (simulated road closures).

### Algorithm:
1. Keep a beam of `B` top configurations
2. At each step, remove one edge and recompute AvgTT/WAvgTT
3. Keep the `B` configurations with the lowest total travel time
4. Repeat for `K` steps

---

## 🧭 Conceptual Intuition

- The model doesn't directly observe which path each car takes — it learns plausible **routing behavior** by minimizing total prediction error
- Early in training, it **explores** many possible routes (high ε or τ)
- Later, it **exploits** the best routes as it gains confidence in travel-time predictions
- Over time, it converges to a realistic equilibrium that resembles **selfish routing** behavior in actual traffic

---

## 🔬 Research Context

This setup bridges:
- **Graph Neural Networks** (for spatial reasoning)
- **Inverse Reinforcement Learning** (for inferring hidden route preferences)
- **Traffic Simulation** (for testing interventions)

---

## 📂 Key Files

| File | Purpose |
|------|----------|
| `gnn_model.py` | Defines GNN architecture and regression head |
| `train_gnn.py` | Implements stochastic candidate route training |
| `exploration_policy.py` | Contains ε-greedy, softmax, Gumbel-Softmax policies |
| `routing.py` | Handles K-shortest paths and path masking |
| `test_gnn_beam.py` | Beam search–based road closure evaluation |

---

## 📖 References

- Maddison et al. (2017). *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.*
- Kool et al. (2019). *Attention, Learn to Solve Routing Problems!*
- Yen, J.Y. (1971). *Finding the K Shortest Loopless Paths in a Network.*
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
- AequilibraE Documentation: [https://aequilibrae.com](https://aequilibrae.com)

---

## ✨ Summary

This GNN combines **spatial reasoning**, **probabilistic routing**, and **traffic equilibrium modeling** into one unified learning system. It learns from trip-level observations alone — discovering both **realistic routes** and **edge-level congestion patterns** — enabling simulations of **road closures**, **traffic rerouting**, and **network resilience**.