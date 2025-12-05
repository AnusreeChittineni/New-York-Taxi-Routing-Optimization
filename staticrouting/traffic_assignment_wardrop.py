"""
NOTE: Code generated with assistance from LLMs. Prompt(s) used:

Create a python script to model wardrop's first principle of traffic assignment, 
using new york taxi trips stored in a duckdb database. As a reminder, wardrop's first
principle states that users route themselves to minimize their trip times, even 
if routes taken by users are not optimal for all users in the system

Perform a simple static traffic assignment on the NYC road network using Wardrop's
first principle (user equilibrium, UE):

- Users (trips) choose routes that minimize their perceived travel times.
- At UE, no user can unilaterally switch to a faster route.

This script:

1. Downloads (or loads from cache) the NYC road network with free-flow travel times.
2. Samples N trips from a NYC TLC taxi DuckDB table and builds an O-D demand set.
3. Runs a link-based user-equilibrium assignment using a basic BPR volume-delay function.
4. Optionally removes a road segment (by OSM ID) from the network before assignment.
5. Reports the average UE travel time across all sampled trips.


Usage example:

    python static_assignment_wardrop.py \
        --duckdb /path/to/nyc_taxi.duckdb \
        --trips_table yellow_2024 \
        --sample_size 5000 \
        --graph_cache ./nyc_graph.graphml

    # With an omitted road (OSM way ID)
    python static_assignment_wardrop.py \
        --duckdb /path/to/nyc_taxi.duckdb \
        --trips_table yellow_2024 \
        --sample_size 5000 \
        --graph_cache ./nyc_graph.graphml \
        --omit_osmid 25161349

Dependencies:
    pip install osmnx networkx duckdb pandas numpy shapely pyproj tqdm pytz
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple, Optional

import duckdb
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm
import pytz

NYC_QUERY = "New York City, New York, USA"
TZ = "America/New_York"

# -----------------------------
# Speed and capacity defaults
# -----------------------------

OSM_SPEED_DEFAULTS_KPH = {
    "motorway": 100,
    "motorway_link": 60,
    "trunk": 80,
    "trunk_link": 50,
    "primary": 60,
    "primary_link": 40,
    "secondary": 50,
    "secondary_link": 35,
    "tertiary": 40,
    "tertiary_link": 30,
    "residential": 30,
    "living_street": 15,
    "unclassified": 35,
    "service": 20,
}

# Very rough lane capacities (veh/h) by highway type
OSM_CAPACITY_DEFAULTS_VPH = {
    "motorway": 2200,
    "motorway_link": 1800,
    "trunk": 2000,
    "trunk_link": 1700,
    "primary": 1800,
    "primary_link": 1500,
    "secondary": 1500,
    "secondary_link": 1200,
    "tertiary": 1200,
    "tertiary_link": 900,
    "residential": 800,
    "living_street": 600,
    "unclassified": 800,
    "service": 400,
}

def parse_maxspeed_to_kph(ms):
    """Parse OSM maxspeed variants to numeric km/h."""
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return None
    if isinstance(ms, (list, tuple)):
        ms = ms[0]
    s = str(ms).lower().strip()
    if any(tok in s for tok in ["signals", "none", "variable"]):
        return None
    num = None
    for token in s.replace(",", " ").split():
        try:
            num = float(token)
            break
        except Exception:
            continue
    if num is None:
        return None
    if "mph" in s:
        return num * 1.60934
    return num

def infer_edge_speed_kph(data) -> float:
    """Return edge free-flow speed_kph using maxspeed or highway defaults."""
    if "maxspeed" in data and data["maxspeed"] is not None:
        sp = parse_maxspeed_to_kph(data["maxspeed"])
        if sp is not None:
            return sp
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0]
    return OSM_SPEED_DEFAULTS_KPH.get(hw, 35.0)

def infer_edge_capacity_vph(data) -> float:
    """Infer a rough per-link capacity (vehicles per hour)."""
    hw = data.get("highway")
    if isinstance(hw, list):
        hw = hw[0]
    base_cap = OSM_CAPACITY_DEFAULTS_VPH.get(hw, 800.0)
    lanes = data.get("lanes")
    if isinstance(lanes, (list, tuple)):
        try:
            lanes = float(lanes[0])
        except Exception:
            lanes = None
    try:
        lanes = float(lanes)
    except Exception:
        lanes = 1.0
    return max(base_cap * lanes, 200.0)


# -----------------------------
# Time helpers
# -----------------------------

def ensure_datetime_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(TZ)


# -----------------------------
# Graph building / loading
# -----------------------------

def build_or_load_graph(cache_path: Optional[str] = None) -> nx.MultiDiGraph:
    """
    Build the NYC drive graph with free-flow travel times and capacity.
    If cache_path exists, load from there; otherwise fetch from OSM and cache.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Graph] Loading cached graph from {cache_path}")
        G = ox.load_graphml(cache_path)
    else:
        print(f"[Graph] Downloading graph for: {NYC_QUERY}")
        G = ox.graph_from_place(NYC_QUERY, network_type="drive", simplify=True)

        # Enrich edges with length, speed, free-flow time, capacity
        for u, v, k, data in G.edges(keys=True, data=True):
            length_m = float(data.get("length", 0.0))
            data["length_m"] = length_m

            sp_kph = float(infer_edge_speed_kph(data))
            data["speed_kph_ff"] = sp_kph

            sp_mps = sp_kph * 1000.0 / 3600.0
            data["t0"] = length_m / max(sp_mps, 1e-6)  # free-flow time (seconds)

            cap = float(infer_edge_capacity_vph(data))
            data["capacity_vph"] = cap

            # Initialize current travel time to free-flow
            data["time"] = data["t0"]
            data["flow"] = 0.0

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            ox.save_graphml(G, cache_path)
            print(f"[Graph] Saved graph to {cache_path}")

    # Ensure numeric types on load
    for u, v, k, data in G.edges(keys=True, data=True):
        for key in ["length_m", "speed_kph_ff", "t0", "capacity_vph"]:
            if key in data:
                try:
                    data[key] = float(data[key])
                except Exception:
                    data[key] = 0.0
        if "time" not in data:
            data["time"] = data.get("t0", 1.0)
        data["flow"] = float(data.get("flow", 0.0))
    return G


# -----------------------------
# Optional: remove a road by OSM ID
# -----------------------------

def remove_road_by_osmid(G: nx.MultiDiGraph, osmid_to_remove: int) -> None:
    """
    Remove all edges whose OSM way ID matches osmid_to_remove.
    Note: OSMnx edges may have osmid as scalar or list.
    """
    print(f"[Graph] Removing road with osmid={osmid_to_remove} (if present)")
    to_remove = []
    for u, v, k, data in G.edges(keys=True, data=True):
        osmids = data.get("osmid")
        if isinstance(osmids, list):
            if osmid_to_remove in osmids:
                to_remove.append((u, v, k))
        else:
            if osmids == osmid_to_remove:
                to_remove.append((u, v, k))

    for (u, v, k) in to_remove:
        G.remove_edge(u, v, key=k)

    print(f"[Graph] Removed {len(to_remove)} edges with osmid={osmid_to_remove}")


# -----------------------------
# Demand: sample trips from DuckDB
# -----------------------------

def sample_od_pairs(
    duckdb_path: str,
    trips_table: str,
    sample_size: int,
    pickup_col: str,
    dropoff_col: str,
    pickup_lat_col: str,
    pickup_lng_col: str,
    dropoff_lat_col: str,
    dropoff_lng_col: str,
    G: nx.MultiDiGraph,
) -> List[Tuple[int, int]]:
    """
    Sample N trips from DuckDB and map them to origin-destination nodes.
    Each sampled trip is treated as demand=1 vehicle from O->D.
    """
    con = duckdb.connect(duckdb_path, read_only=True)
    print(f"[Demand] Sampling {sample_size} trips from {trips_table}")

    # Using DuckDB's TABLESAMPLE for random sampling
    query = f"""
        SELECT
            {pickup_lat_col}  AS pu_lat,
            {pickup_lng_col}  AS pu_lng,
            {dropoff_lat_col} AS do_lat,
            {dropoff_lng_col} AS do_lng,
            {pickup_col}      AS pu_time,
            {dropoff_col}     AS do_time
        FROM {trips_table}
        USING SAMPLE {sample_size} ROWS;
    """
    df = con.execute(query).fetch_df()
    print(f"[Demand] Sampled {len(df)} raw trips")

    # Clean
    df = df.dropna(
        subset=["pu_lat", "pu_lng", "do_lat", "do_lng", "pu_time", "do_time"]
    ).copy()
    df["pu_time"] = ensure_datetime_series(df["pu_time"])
    df["do_time"] = ensure_datetime_series(df["do_time"])
    df["trip_seconds"] = (df["do_time"] - df["pu_time"]).dt.total_seconds()
    df = df[(df["trip_seconds"] > 60) & (df["trip_seconds"] < 3 * 3600)].copy()

    if df.empty:
        raise RuntimeError("No valid trips after cleaning")

    # Map to nearest graph nodes
    xs_pu = df["pu_lng"].to_numpy()
    ys_pu = df["pu_lat"].to_numpy()
    xs_do = df["do_lng"].to_numpy()
    ys_do = df["do_lat"].to_numpy()

    pu_nodes = ox.distance.nearest_nodes(G, xs_pu, ys_pu)
    do_nodes = ox.distance.nearest_nodes(G, xs_do, ys_do)

    df["o_node"] = pu_nodes
    df["d_node"] = do_nodes

    # Build list of OD pairs (one trip per row)
    od_pairs = list(zip(df["o_node"].astype(int), df["d_node"].astype(int)))
    print(f"[Demand] Built {len(od_pairs)} OD demands")

    return od_pairs


# -----------------------------
# User equilibrium assignment
# -----------------------------

def bpr_travel_time(t0: float, flow_vph: float, capacity_vph: float, alpha=0.15, beta=4.0) -> float:
    """
    Bureau of Public Roads (BPR) volume-delay function:
        t = t0 * (1 + alpha * (v/c)^beta)
    """
    if capacity_vph <= 0:
        return t0
    x = max(flow_vph / capacity_vph, 0.0)
    return t0 * (1.0 + alpha * (x ** beta))


def all_or_nothing_assignment(
    G: nx.MultiDiGraph, od_pairs: List[Tuple[int, int]], weight_attr: str = "time"
) -> Dict[Tuple[int, int, int], float]:
    """
    Perform an all-or-nothing (AON) assignment:
    For each OD pair, place all flow on the current shortest path.
    Returns a dict (u, v, k) -> flow (veh/h), assuming each OD = 1 veh/h.
    """
    aux_flow = {(u, v, k): 0.0 for (u, v, k) in G.edges(keys=True)}
    for (o, d) in od_pairs:
        try:
            path = nx.shortest_path(G, source=o, target=d, weight=weight_attr, method="dijkstra")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for u, v in zip(path[:-1], path[1:]):
            # choose key with minimal current weight
            keys = list(G[u][v].keys())
            if not keys:
                continue
            k = min(
                keys, key=lambda kk: G[u][v][kk].get(weight_attr, G[u][v][kk].get("t0", 1.0))
            )
            aux_flow[(u, v, k)] += 1.0  # 1 vehicle per OD
    return aux_flow


def user_equilibrium_assignment(
    G: nx.MultiDiGraph,
    od_pairs: List[Tuple[int, int]],
    max_iters: int = 20,
    tol: float = 1e-3,
) -> None:
    """
    Perform a simple UE assignment using Method of Successive Averages (MSA).

    - Initialize flows with an AON on free-flow times (t0).
    - Repeat:
        - update link travel times with BPR function
        - compute AON on current times
        - MSA update of flows

    Flows are stored in G[u][v][k]['flow'] (veh/h).
    Current travel times are in G[u][v][k]['time'] (seconds).
    """
    print("[UE] Initial all-or-nothing on free-flow times")
    # Initialize flows at zero
    for u, v, k, data in G.edges(keys=True, data=True):
        data["flow"] = 0.0
        data["time"] = float(data.get("t0", 1.0))

    # First AON on free-flow
    aux_flow = all_or_nothing_assignment(G, od_pairs, weight_attr="t0")
    for (u, v, k), f in aux_flow.items():
        G[u][v][k]["flow"] = f

    for it in range(1, max_iters + 1):
        print(f"[UE] Iteration {it}/{max_iters}")

        # 1) Update travel times with BPR
        for u, v, k, data in G.edges(keys=True, data=True):
            t0 = float(data.get("t0", 1.0))
            cap = float(data.get("capacity_vph", 800.0))
            flow = float(data.get("flow", 0.0))
            data["time"] = bpr_travel_time(t0, flow, cap)

        # 2) AON on current times
        aux_flow = all_or_nothing_assignment(G, od_pairs, weight_attr="time")

        # 3) MSA step: flow^{k+1} = flow^k + (1/k) * (aux - flow^k)
        step = 1.0 / float(it + 1)  # can also use 1/it; we start from it=1 after initial
        max_delta = 0.0
        for u, v, k, data in G.edges(keys=True, data=True):
            f_old = float(data.get("flow", 0.0))
            f_aux = aux_flow.get((u, v, k), 0.0)
            f_new = f_old + step * (f_aux - f_old)
            data["flow"] = f_new
            max_delta = max(max_delta, abs(f_new - f_old))

        print(f"[UE]   max flow change = {max_delta:.4f}")
        if max_delta < tol:
            print("[UE] Converged under tolerance")
            break


# -----------------------------
# Compute average UE travel time for sampled trips
# -----------------------------

def compute_average_travel_time(
    G: nx.MultiDiGraph, od_pairs: List[Tuple[int, int]], weight_attr: str = "time"
) -> float:
    """
    Compute the average travel time across all OD pairs using current edge times.
    """
    total_time = 0.0
    count = 0
    for (o, d) in od_pairs:
        try:
            path = nx.shortest_path(G, source=o, target=d, weight=weight_attr, method="dijkstra")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        t = 0.0
        for u, v in zip(path[:-1], path[1:]):
            keys = list(G[u][v].keys())
            if not keys:
                continue
            k = min(
                keys, key=lambda kk: G[u][v][kk].get(weight_attr, G[u][v][kk].get("t0", 1.0))
            )
            t += float(G[u][v][k].get(weight_attr, 0.0))
        total_time += t
        count += 1
    if count == 0:
        return float("nan")
    return total_time / count


# -----------------------------
# Main CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Static UE assignment on NYC using Wardrop's first principle")
    ap.add_argument("--duckdb", required=True, help="Path to DuckDB file with NYC TLC trips")
    ap.add_argument("--trips_table", required=True, help="Trip table name in DuckDB (e.g., yellow_2024)")
    ap.add_argument("--sample_size", type=int, default=5000, help="Number of trips to sample for OD demand")
    ap.add_argument("--pickup_col", default="tpep_pickup_datetime", help="Pickup datetime column")
    ap.add_argument("--dropoff_col", default="tpep_dropoff_datetime", help="Dropoff datetime column")
    ap.add_argument("--pickup_lat_col", default="pickup_latitude", help="Pickup latitude column")
    ap.add_argument("--pickup_lng_col", default="pickup_longitude", help="Pickup longitude column")
    ap.add_argument("--dropoff_lat_col", default="dropoff_latitude", help="Dropoff latitude column")
    ap.add_argument("--dropoff_lng_col", default="dropoff_longitude", help="Dropoff longitude column")
    ap.add_argument("--graph_cache", default="./nyc_ue_graph.graphml", help="Path to cache the OSMnx graph")
    ap.add_argument("--omit_osmid", type=int, default=None, help="Optional OSM way id of road to remove")
    ap.add_argument("--max_iters", type=int, default=20, help="Maximum UE iterations")
    ap.add_argument("--tol", type=float, default=1e-3, help="Flow-change convergence tolerance")
    args = ap.parse_args()

    # 1. Build/load graph
    G = build_or_load_graph(args.graph_cache)

    # 2. Optionally remove an OSM way from the graph
    if args.omit_osmid is not None:
        remove_road_by_osmid(G, args.omit_osmid)

    # 3. Build OD demands from sampled trips
    od_pairs = sample_od_pairs(
        duckdb_path=args.duckdb,
        trips_table=args.trips_table,
        sample_size=args.sample_size,
        pickup_col=args.pickup_col,
        dropoff_col=args.dropoff_col,
        pickup_lat_col=args.pickup_lat_col,
        pickup_lng_col=args.pickup_lng_col,
        dropoff_lat_col=args.dropoff_lat_col,
        dropoff_lng_col=args.dropoff_lng_col,
        G=G,
    )

    # 4. Run UE assignment
    user_equilibrium_assignment(G, od_pairs, max_iters=args.max_iters, tol=args.tol)

    # 5. Compute average UE travel time
    avg_time_s = compute_average_travel_time(G, od_pairs, weight_attr="time")
    if math.isnan(avg_time_s):
        print("Average UE travel time: NaN (no usable OD paths)")
    else:
        print(f"Average UE travel time at user equilibrium (over {len(od_pairs)} trips): {avg_time_s:.2f} seconds")

if __name__ == "__main__":
    main()

