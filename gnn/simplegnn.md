### Purpose
This project builds a **Graph Neural Network (GNN)** that learns how traffic flows through Manhattan using taxi trip data.  
It predicts how long it takes to travel along different roads and how traffic patterns change if certain roads are closed.

---

### What the Model Does
1. **Takes in the road network of Manhattan** — intersections as nodes, roads as edges.
2. **Predicts travel time for each road** using features like:
   - road length  
   - capacity  
   - free-flow speed  
   - current traffic (flow)
3. **Learns from real taxi trips** — given a start and end point plus the actual trip time.
4. **Figures out which routes are likely** by trying multiple candidate routes and learning which are most consistent with real-world travel times.
5. **Simulates road closures** — tests how removing certain edges affects travel time across the city.

---

### How It Learns
Since we don’t know which exact route each taxi took, the GNN uses **exploration**:
- It tries **multiple possible paths** for each trip (K-shortest paths).
- At first, it explores randomly to learn different routes.
- Over time, it focuses more on the routes that best match observed travel times.
- This is similar to **reinforcement learning**, but trained using travel-time error as feedback.

---

### Training Objective
For every trip:
1. The GNN predicts travel times for all roads.
2. For each candidate path between the origin and destination, it adds up the edge times to estimate a total trip time.
3. It compares the predicted trip time to the real one and adjusts weights to make predictions more accurate.
4. The model also keeps some randomness (exploration) so it doesn’t get stuck always picking the same path too early.

---

### After Training
Once trained, the model can:
- Estimate travel times between any two points.
- Predict how the average travel time changes if you **close or remove** certain roads.
- Be used to find which roads are **critical** or **redundant** for traffic flow.

---

### Summary
In simple terms:
- The GNN learns **how fast traffic moves** on each road.  
- It **figures out routes** that fit observed travel times without being told the actual paths.  
- It can **simulate closures** to see which roads matter most.

Think of it as building a data-driven **virtual Manhattan** where you can test how traffic would react to any change.
