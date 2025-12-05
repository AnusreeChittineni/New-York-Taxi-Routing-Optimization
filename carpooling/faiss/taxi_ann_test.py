import duckdb
import pandas as pd
import numpy as np
import faiss
from geopy.distance import geodesic
from datetime import datetime

# =====================================================
# Load data from DuckDB
# =====================================================

print("Loading data from DuckDB...")

con = duckdb.connect("../../data/duckdb/nyc_routing.duckdb")  # change if your file is different

df = con.execute("""
    SELECT 
        pickup_lat,
        pickup_lon,
        tpep_pickup_datetime,
        passenger_count
    FROM taxi_clean
    LIMIT 500000               -- adjust if needed
""").df()

print(f"Loaded {len(df):,} rows")

# Convert timestamps
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

# =====================================================
# Prepare FAISS input vectors
# =====================================================

print("Preparing FAISS vectors...")

points = df[["pickup_lat", "pickup_lon"]].astype("float32").to_numpy()

d = 2  # dimensions: (lat, lon)
index = faiss.IndexFlatL2(d)

print("Building FAISS index...")
index.add(points)

print(f"FAISS index contains {index.ntotal:,} vectors")

# =====================================================
# Filtering function
# =====================================================

def neighbors_with_filters(
    index,
    points,
    df,
    idx,
    radius_meters,
    time_window_minutes,
    max_passengers=4,
    k=200
):
    q = points[idx].reshape(1, -1)
    distances, ids = index.search(q, k)

    results = []
    t0 = df.iloc[idx]["tpep_pickup_datetime"]

    p1 = (df.iloc[idx]["pickup_lat"], df.iloc[idx]["pickup_lon"])

    for neighbor_idx in ids[0]:
        if neighbor_idx < 0 or neighbor_idx == idx:
            continue

        # Distance (geodesic)
        p2 = (df.iloc[neighbor_idx]["pickup_lat"], df.iloc[neighbor_idx]["pickup_lon"])
        dist_m = geodesic(p1, p2).meters
        if dist_m > radius_meters:
            continue

        # Time window
        t1 = df.iloc[neighbor_idx]["tpep_pickup_datetime"]
        delta_min = abs((t1 - t0).total_seconds()) / 60
        if delta_min > time_window_minutes:
            continue

        # Passenger limit
        if df.iloc[neighbor_idx]["passenger_count"] > max_passengers:
            continue

        results.append((neighbor_idx, dist_m, delta_min))

    return results

# =====================================================
# Parameter sweep
# =====================================================

radii = [100, 250, 500, 1000]   # meters
time_windows = [1, 3, 5, 10]    # minutes
max_passengers = 4

print("\nStarting ANN radius/time sweep...\n")

stats = []

for r in radii:
    for tw in time_windows:
        print(f"Testing radius={r} m, time_window={tw} min ...")

        total_matches = 0

        for i in range(len(df)):
            neighbors = neighbors_with_filters(
                index=index,
                points=points,
                df=df,
                idx=i,
                radius_meters=r,
                time_window_minutes=tw,
                max_passengers=max_passengers,
                k=200,
            )
            total_matches += len(neighbors)

        stats.append((r, tw, total_matches))
        print(f"Matches found: {total_matches:,}\n")

print("Sweep completed.\nResults:")
for r, tw, m in stats:
    print(f"Radius {r} m | Time {tw} min | Matches {m:,}")
