"""
build_nyc_gnn_graph.py

Builds a time-series road graph for NYC suitable for training spatio-temporal
graph neural networks. It performs four key steps:

1) Download the NYC road network (nodes = intersections, edges = road segments)
   with attributes (length, inferred speed_kph, oneway) using OSMnx.

2) Aggregate NYC TLC trips (stored in a DuckDB database) into a time-dependent
   O-D demand tensor D(t) over discrete bins (e.g., 15 minutes) AFTER snapping
   pickups/dropoffs to nearest road nodes.

3) Infer a most-likely path for each trip using either:
   - OSRM (if --osrm_url is provided): time-dependent edge weights if you pass a
     profile that reflects historical speeds, or
   - NetworkX shortest path with per-edge costs derived from *historical* average
     speeds per time bin. On first pass, we fall back to free-flow speeds inferred
     from OSM (maxspeed or highway defaults).

4) From each trip’s observed duration (dropoff - pickup), distribute per-link
   travel times along the chosen path and aggregate into a per-edge, per-time-bin
   feature matrix (counts, mean speed, mean travel time, congestion factor).
taxi_data
Outputs:
- graph.graphml: the road graph with canonicalized attributes
- od_matrix.parquet: sparse O-D counts per time-bin
- link_features.parquet: per-edge, per-time-bin aggregated features
- node_map.parquet: lookup from internal node id -> (osmid, x, y)
- edge_map.parquet: lookup from internal edge id -> (u, v, key, osmid, length_m)

Notes:
- You must supply a DuckDB file with a trips table. By default we assume NYC Yellow Taxi
  schema columns: pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude,
  tpep_pickup_datetime, tpep_dropoff_datetime. You can override with CLI args.
- Timezone is America/New_York.
- For reproducibility and scale, we operate in chunks over the DuckDB table.

Example:
    python build_nyc_gnn_graph.py \
        --duckdb /path/nyc_taxi.duckdb --trips_table taxi_data \
        --time_bin_minutes 15 --osrm_url http://localhost:5000 \
        --output_dir ./nyc_gnn_data

Dependencies:
    pip install osmnx networkx duckdb pandas numpy pyarrow shapely pyproj tqdm requests pytz
"""
from __future__ import annotations

import argparse
import os
import json
import math
import pdb
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import duckdb
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pytz
import requests
from shapely.geometry import Point
from tqdm import tqdm

# --------------------------
# Helpers & Defaults
# --------------------------

NYC_QUERY = "New York City, New York, USA"
TZ = "America/New_York"

# Fallback free-flow speeds by highway tag (kph), conservative
OSM_SPEED_DEFAULTS_KPH = {
    "motorway": 100, "motorway_link": 60,
    "trunk": 80, "trunk_link": 50,
    "primary": 60, "primary_link": 40,
    "secondary": 50, "secondary_link": 35,
    "tertiary": 40, "tertiary_link": 30,
    "residential": 30, "living_street": 15,
    "unclassified": 35, "service": 20
}

def parse_maxspeed_to_kph(ms):
    """Parse OSM 'maxspeed' variants into numeric kph."""
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return None
    # maxspeed can be list-like, str like '25 mph', '50', 'signals', etc.
    if isinstance(ms, (list, tuple)):
        ms = ms[0]
    s = str(ms).lower().strip()
    # Discard non-numeric maxspeed markers
    if any(tok in s for tok in ["signals", "none", "variable"]):
        return None
    # Extract number
    num = None
    for token in s.replace(",", " ").split():
        try:
            num = float(token)
            break
        except Exception:
            continue
    if num is None:
        return None
    # unit detection
    if "mph" in s:
        return num * 1.60934
    # assume km/h otherwise
    return num

def infer_edge_speed_kph(data):
    """Return edge free-flow speed_kph using maxspeed or highway defaults."""
    if 'maxspeed' in data and data['maxspeed'] is not None:
        sp = parse_maxspeed_to_kph(data['maxspeed'])
        if sp is not None:
            return sp
    hw = data.get('highway')
    if isinstance(hw, list):
        hw = hw[0]
    return OSM_SPEED_DEFAULTS_KPH.get(hw, 35.0)

def ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Robustly parse datetimes and localize to America/New_York."""
    dt = pd.to_datetime(s, errors='coerce', utc=True)
    # Localize to NYC
    return dt.dt.tz_convert(TZ)

def to_time_bin(dt: pd.Series, bin_minutes: int) -> pd.Series:
    """Floor to time bin in local time; return naive timestamp key (UTC) for grouping."""
    # Already tz-aware NYC; floor and convert to UTC for consistent indexing
    binned_local = dt.dt.floor(f"{bin_minutes}min")
    # Convert to UTC key for storage, but keep naive timestamp for file compactness
    binned_utc = binned_local.dt.tz_convert("UTC").dt.tz_localize(None)
    return binned_utc

def nearest_nodes_cached(G, xs, ys, batch=10000):
    """Vectorized nearest node lookup with batching to limit memory."""
    # osmnx.nearest_nodes supports vectorized coords
    out = []
    for i in range(0, len(xs), batch):
        out.extend(ox.distance.nearest_nodes(G, X=xs[i:i+batch], Y=ys[i:i+batch]))
    return np.array(out)

@dataclass
class Args:
    duckdb: str
    trips_table: str
    pickup_col: str
    dropoff_col: str
    pickup_lat_col: str
    pickup_lng_col: str
    dropoff_lat_col: str
    dropoff_lng_col: str
    trip_distance_col: str
    time_bin_minutes: int
    output_dir: str
    osrm_url: Optional[str]
    max_rows: Optional[int]
    chunk_size: int

# --------------------------
# Step 1: Build/Download Graph
# --------------------------

def build_road_graph(place: str = NYC_QUERY, cache_path: str | None = None) -> nx.MultiDiGraph:
    print(f"[Step 1] Downloading OSM road network for: {place}")

    if cache_path and os.path.exists(cache_path):
        print(f"[Step 1] Loading cached road graph from {cache_path}")
        G = ox.load_graphml(cache_path)
        print(f"[Step 1] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # simpify the graph by combining the segments for a road into a single edge
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    # Ensure required attributes
    for u, v, k, data in G.edges(keys=True, data=True):
        data['length_m'] = float(data.get('length', 0.0))
        data['oneway'] = bool(data.get('oneway', False))
        data['speed_kph_ff'] = float(infer_edge_speed_kph(data))
        # cost placeholder: seconds at free-flow
        sp_mps = data['speed_kph_ff'] * 1000.0 / 3600.0
        data['travel_time_ff_s'] = data['length_m'] / max(sp_mps, 1e-6)
    # Canonical node attributes
    for n, nd in G.nodes(data=True):
        nd['x'] = float(nd.get('x'))
        nd['y'] = float(nd.get('y'))
    print(f"[Step 1] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    ox.save_graphml(G, graphml_path)
    return G

# --------------------------
# Step 2: Build time-dependent O-D demand
# --------------------------

def load_and_bin_trips(con: duckdb.DuckDBPyConnection, args: Args, G: nx.MultiDiGraph) -> pd.DataFrame:
    print(f"[Step 2] Reading trips from DuckDB table: {args.trips_table}")
    # Select minimal columns
    query = f"""
        SELECT
            {args.pickup_lat_col} AS pu_lat,
            {args.pickup_lng_col} AS pu_lng,
            {args.dropoff_lat_col} AS do_lat,
            {args.dropoff_lng_col} AS do_lng,
            {args.trip_distance_col} AS trip_distance_miles,
            {args.pickup_col} AS pu_time,
            {args.dropoff_col} AS do_time
        FROM {args.trips_table}
    """
    if args.max_rows:
        query += f" LIMIT {int(args.max_rows)}"
    df = con.execute(query).fetch_df()
    print(f"[Step 2] Loaded {len(df):,} trips")
    # Clean rows
    df = df.dropna(subset=['pu_lat','pu_lng','do_lat','do_lng','pu_time','do_time']).copy()
    # Parse datetimes and filter negative/zero durations
    df['pu_time'] = ensure_datetime_series(df['pu_time'])
    df['do_time'] = ensure_datetime_series(df['do_time'])
    df['trip_seconds'] = (df['do_time'] - df['pu_time']).dt.total_seconds()
    df = df[(df['trip_seconds'] > 60) & (df['trip_seconds'] < 3*3600)].copy()

    # get trip distance for more accurate road speed estimations 
    df['trip_distance_km'] = df['trip_distance_miles'].astype(float) * 1.60934

    # Snap to graph nodes
    pu_nodes = nearest_nodes_cached(G, df['pu_lng'].to_numpy(), df['pu_lat'].to_numpy())
    do_nodes = nearest_nodes_cached(G, df['do_lng'].to_numpy(), df['do_lat'].to_numpy())
    df['o_node'] = pu_nodes
    df['d_node'] = do_nodes
    # Time bin
    df['time_bin'] = to_time_bin(df['pu_time'], args.time_bin_minutes)
    # OD demand counts
    od = (df.groupby(['time_bin','o_node','d_node'])
            .size()
            .rename('count')
            .reset_index())
    print(f"[Step 2] Built OD counts with {len(od):,} (time_bin, O, D) entries")
    return df, od

# --------------------------
# Step 3: Path inference
# --------------------------

def route_with_osrm(osrm_url: str, src_xy: Tuple[float,float], dst_xy: Tuple[float,float]) -> Optional[List[Tuple[float,float]]]:
    """Call OSRM route service; return list of (lon, lat) coordinates for the path geometry if available."""
    url = f"{osrm_url.rstrip('/')}/route/v1/driving/{src_xy[0]},{src_xy[1]};{dst_xy[0]},{dst_xy[1]}"
    params = {"overview":"full","geometries":"geojson","steps":"false","annotations":"false"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("routes"):
            coords = data['routes'][0]['geometry']['coordinates']
            return [(float(lon), float(lat)) for lon, lat in coords]
    except Exception as e:
        return None
    return None

def shortest_path_nx(G: nx.MultiDiGraph, o: int, d: int, weight_attr: str = "travel_time_ff_s") -> Optional[List[int]]:

    def w(u, v, data):
        val = data.get(weight_attr, 1.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            # Fallback to something finite
            return 1.0
    
    try:
        path = nx.shortest_path(G, source=o, target=d, weight=w, method='dijkstra')
        return path
    except Exception:
        return None

def mapmatch_coords_to_edges(G: nx.MultiDiGraph, coords: List[Tuple[float,float]]) -> Optional[List[Tuple[int,int,int]]]:
    """
    Very lightweight snapping of a polyline to nearest edges sequence.
    Returns list of (u, v, key) for edges approximating the path.
    """
    if not coords: return None
    nodes = ox.distance.nearest_nodes(G, X=[c[0] for c in coords], Y=[c[1] for c in coords])
    # Build consecutive edges
    edges = []
    for i in range(len(nodes)-1):
        u, v = nodes[i], nodes[i+1]
        if G.has_edge(u, v):
            # choose shortest key
            keys = list(G[u][v].keys())
            key = min(keys, key=lambda k: G[u][v][k].get('length_m', 1e9))
            edges.append((u, v, key))
        else:
            # try shortest path fill
            sp = shortest_path_nx(G, u, v)
            if sp and len(sp) > 1:
                for s, t in zip(sp[:-1], sp[1:]):
                    keys = list(G[s][t].keys())
                    key = min(keys, key=lambda k: G[s][t][k].get('length_m', 1e9))
                    edges.append((s, t, key))
    return edges or None

def path_edges_from_nodes(G: nx.MultiDiGraph, path_nodes: List[int]) -> List[Tuple[int,int,int]]:
    edges = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        keys = list(G[u][v].keys())
        key = min(keys, key=lambda k: G[u][v][k].get('length_m', 1e9))
        edges.append((u, v, key))
    return edges

# --------------------------
# Step 4: Per-link time allocation & aggregation
# --------------------------

def allocate_link_times(G: nx.MultiDiGraph, edges_seq: list[tuple[int, int, int]], total_trip_seconds: float):
    """
    Allocate observed trip time across edges using the given formula:
        t_li = T_li_free + (T_li_free / Σ_j T_lj_free) * (T_observed - T_freeflow)

    Args:
        G: networkx.MultiDiGraph
        edges_seq: list of (u, v, key) tuples for the path
        total_trip_seconds: observed trip duration (T_observed)

    Returns:
        list of (u, v, key, allocated_time_s)
    """
    if not edges_seq:
        return []

    # Get free-flow travel times for all edges in the path
    free_times = np.array(
        [max(float(G[u][v][k].get('travel_time_ff_s', 0.0)), 1e-6) for (u, v, k) in edges_seq],
        dtype=float
    )

    # Total free-flow time for the trip
    T_freeflow = free_times.sum()

    # Compute the total observed minus free-flow difference
    delta_T = total_trip_seconds - T_freeflow

    # Apply the formula
    allocated_times = free_times + (free_times / T_freeflow) * delta_T

    # Prevent negative allocations (in case observed < free-flow)
    allocated_times = np.clip(allocated_times, a_min=0.0, a_max=None)

    return [(u, v, k, t) for (u, v, k), t in zip(edges_seq, allocated_times.tolist())]


# --------------------------
# Main pipeline
# --------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duckdb", required=True, help="Path to DuckDB file with NYC TLC trips")
    p.add_argument("--trips_table", required=True, help="Table name with trips")
    p.add_argument("--pickup_col", default="tpep_pickup_datetime")
    p.add_argument("--dropoff_col", default="tpep_dropoff_datetime")
    p.add_argument("--pickup_lat_col", default="pickup_latitude")
    p.add_argument("--pickup_lng_col", default="pickup_longitude")
    p.add_argument("--dropoff_lat_col", default="dropoff_latitude")
    p.add_argument("--dropoff_lng_col", default="dropoff_longitude")
    p.add_argument("--trip_distance_col", default="trip_distance")
    p.add_argument("--time_bin_minutes", type=int, default=15)
    p.add_argument("--output_dir", default="./nyc_gnn_data")
    p.add_argument("--osrm_url", default=None, help="Optional OSRM base URL, e.g., http://localhost:5000")
    p.add_argument("--max_rows", type=int, default=None, help="Optional limit for quick runs")
    p.add_argument("--chunk_size", type=int, default=200000, help="Process trips in chunks when writing features")
    args_ns = p.parse_args()

    args = Args(**vars(args_ns))

    os.makedirs(args.output_dir, exist_ok=True)

    # Persist base graph
    graphml_path = os.path.join(args.output_dir, "graph.graphml")
    # Step 1
    G = build_road_graph(NYC_QUERY, cache_path=graphml_path)

    # Node/edge lookup tables
    node_map = pd.DataFrame(
        [{'node_id': n, 'osmid': G.nodes[n].get('osmid', n), 'x': G.nodes[n]['x'], 'y': G.nodes[n]['y']}
         for n in G.nodes()]
    )
    # Edge id: enumerate stable index
    edge_rows = []
    for i, (u,v,k,data) in enumerate(G.edges(keys=True, data=True)):
        edge_rows.append({
            'edge_ix': i,
            'u': u, 'v': v, 'key': k,
  #          'osmid': data.get('osmid'),
            'length_m': data.get('length_m', np.nan)
        })
    edge_map = pd.DataFrame(edge_rows)

    node_map.to_parquet(os.path.join(args.output_dir, "node_map.parquet"))
    edge_map.to_parquet(os.path.join(args.output_dir, "edge_map.parquet"))

    # Step 2
    con = duckdb.connect(args.duckdb, read_only=True)
    trips_df, od_df = load_and_bin_trips(con, args, G)
    od_df.to_parquet(os.path.join(args.output_dir, "od_matrix.parquet"))

    # Prepare quick index for node->coords for OSRM routing
    node_xy = node_map.set_index('node_id')[['x','y']].to_dict('index')

    # Containers for per-edge, per-time-bin aggregation
    agg_records = []  # dicts with time_bin, edge_ix, count, sum_time, sum_len, etc.

    # Optional pre-compute a mapping from (u,v,k) -> edge_ix
    uvk_to_ix = {(row.u, row.v, row.key): int(row.edge_ix) for row in edge_map.itertuples(index=False)}

    # Iterate trips; for scalability, we chunk to limit memory during path inference
    print(f"[Step 3-4] Inferring paths and aggregating per-link features ...")
    # Sort by time_bin for locality
    trips_df = trips_df.sort_values('time_bin')

    # Helper to compute path edges for one trip
    def infer_edges_for_trip(row) -> Optional[List[Tuple[int,int,int]]]:
        o, d = int(row['o_node']), int(row['d_node'])
        if args.osrm_url:
            # OSRM path geometry -> mapmatch to edges
            src = (node_xy[o]['x'], node_xy[o]['y'])
            dst = (node_xy[d]['x'], node_xy[d]['y'])
            coords = route_with_osrm(args.osrm_url, src, dst)
            if coords:
                edges_seq = mapmatch_coords_to_edges(G, coords)
                if edges_seq:
                    return edges_seq
        # Fallback: shortest path by free-flow travel time
        path_nodes = shortest_path_nx(G, o, d, weight_attr='travel_time_ff_s')
        if not path_nodes:
            return None
        return path_edges_from_nodes(G, path_nodes)

    # Process in chunks to avoid building a huge list in memory
    it = range(0, len(trips_df), args.chunk_size)
    for start in tqdm(it, total=math.ceil(len(trips_df)/args.chunk_size)):
        chunk = trips_df.iloc[start:start+args.chunk_size].copy()
        # Infer paths
        paths = []
        for row in chunk.itertuples(index=False):
            edges_seq = infer_edges_for_trip(row._asdict())
            paths.append(edges_seq)

        # Allocate times and write aggregation records
        for r, edges_seq in zip(chunk.itertuples(index=False), paths):
            if not edges_seq:
                continue

            trip_dist_m = float(r.trip_distance_km) * 1000.0
            graph_len_m = 0.0
            for (u, v, k) in edges_seq:
                graph_len_m += float(G[u][v][k].get('length_m', 0.0))
            
            # Scale factor so that sum(edge_length_scaled) = real TLC distance
            scale_factor = trip_dist_m / graph_len_m

            alloc = allocate_link_times(G, edges_seq, float(r.trip_seconds))
            tb = r.time_bin
            for (u,v,k, tsec) in alloc:
                edge_ix = uvk_to_ix.get((u,v,k))
                if edge_ix is None:
                    continue
                length_m_graph = G[u][v][k].get('length_m', np.nan)
                length_m_scaled = float(length_m_graph) * scale_factor
                speed_kph_ff = G[u][v][k].get('speed_kph_ff', np.nan)
                agg_records.append({
                    'time_bin': tb,
                    'edge_ix': edge_ix,
                    'u': u, 'v': v, 'key': k,
                    'alloc_time_s': tsec,
                    'length_m_graph': float(length_m_graph),
                    'length_m_scaled': float(length_m_scaled),
                    'speed_kph_ff': float(speed_kph_ff)
                })

        # Periodically flush to disk if very large (optional)

    # Aggregate per-edge per-time_bin
    if len(agg_records) == 0:
        print("WARNING: No aggregation records created. Check inputs.")
        link_features = pd.DataFrame(columns=[
            'time_bin','edge_ix','count','mean_travel_time_s','mean_speed_kph','congestion_factor'
        ])
    else:
        agg_df = pd.DataFrame(agg_records)
        agg_df['count'] = 1
        grouped = agg_df.groupby(['time_bin','edge_ix'])
        # Average per-link travel time = sum(alloc_time)/count
        feats = grouped.agg(
            total_alloc_time_s=('alloc_time_s','sum'),
            total_length_m_graph=('length_m_graph','sum'),
            total_length_m_scaled=('length_m_scaled', 'sum'),
            count=('count','sum'),
            speed_kph_ff=('speed_kph_ff','mean')
        ).reset_index()
        feats['mean_travel_time_s'] = feats['total_alloc_time_s'] / feats['count'].clip(lower=1)
        # Mean observed speed along edges in the trip allocation
        # Using total_length / total_time * 3.6
        feats['mean_speed_kph'] = (feats['total_length_m_scaled'] / feats['total_alloc_time_s'].clip(lower=1e-6)) * 3.6
        # Congestion factor = freeflow_speed / observed_speed (>= ~1 means slower than free flow)
        feats['congestion_factor'] = feats['speed_kph_ff'] / feats['mean_speed_kph'].replace(0, np.nan)
        link_features = feats[['time_bin','edge_ix','count','mean_travel_time_s','mean_speed_kph','congestion_factor']]
    
    breakpoint()
    link_features_path = os.path.join(args.output_dir, "link_features.parquet")
    link_features.to_parquet(link_features_path)

    # Save a small JSON metadata with parameter choices
    meta = {
        "place": NYC_QUERY,
        "time_bin_minutes": args.time_bin_minutes,
        "tz": TZ,
        "osrm_url": args.osrm_url,
        "trips_table": args.trips_table,
        "features": list(link_features.columns),
        "graphml": os.path.abspath(graphml_path)
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("[Done] Wrote:")
    print(f"  - {graphml_path}")
    print(f"  - {os.path.join(args.output_dir, 'od_matrix.parquet')}")
    print(f"  - {link_features_path}")
    print(f"  - {os.path.join(args.output_dir, 'node_map.parquet')}")
    print(f"  - {os.path.join(args.output_dir, 'edge_map.parquet')}")
    print(f"  - {os.path.join(args.output_dir, 'metadata.json')}")

if __name__ == "__main__":
    main()
