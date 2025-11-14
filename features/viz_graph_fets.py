"""

Analyze and visualize graph-level traffic features

Given a directory with:
  - graph.graphml          (OSMnx road graph)
  - edge_map.parquet       (edge_ix -> (u, v, key, length_m))
  - link_features.parquet  (time_bin, edge_ix, count, mean_travel_time_s,
                            mean_speed_kph, congestion_factor)

This script will:

1. Report:
   - number of nodes and edges
   - global stats for congestion_factor and mean_speed_kph
2. Plot:
   - histogram of congestion_factor
   - a "heatmap" of the road network, colored by a chosen metric

Usage:

    python analyze_nyc_graph_features.py \
        --data_dir ./nyc_gnn_data \
        --metric congestion_factor \
        --output_prefix nyc_stats

Requirements:
    pip install osmnx pandas numpy pyarrow matplotlib
"""

from __future__ import annotations

import argparse
import os
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import osmnx as ox


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing graph.graphml, edge_map.parquet, link_features.parquet",
    )
    p.add_argument(
        "--metric",
        default="congestion_factor",
        choices=["congestion_factor", "mean_speed_kph", "mean_travel_time_s", "count"],
        help="Edge feature to visualize in the heatmap",
    )
    p.add_argument(
        "--time_start",
        default=None,
        help="Optional lower bound on time_bin (inclusive), e.g. '2024-01-01 00:00:00'",
    )
    p.add_argument(
        "--time_end",
        default=None,
        help="Optional upper bound on time_bin (exclusive), e.g. '2024-01-02 00:00:00'",
    )
    p.add_argument(
        "--output_prefix",
        default="graph_report",
        help="Prefix for output PNG files",
    )
    return p.parse_args()


def main():
    args = parse_args()

    graph_path = os.path.join(args.data_dir, "graph.graphml")
    edge_map_path = os.path.join(args.data_dir, "edge_map.parquet")
    link_features_path = os.path.join(args.data_dir, "link_features.parquet")

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"graph.graphml not found at {graph_path}")
    if not os.path.exists(edge_map_path):
        raise FileNotFoundError(f"edge_map.parquet not found at {edge_map_path}")
    if not os.path.exists(link_features_path):
        raise FileNotFoundError(f"link_features.parquet not found at {link_features_path}")

    print(f"[Load] Reading graph from {graph_path}")
    G = ox.load_graphml(graph_path)

    print(f"[Load] Reading edge_map from {edge_map_path}")
    edge_map = pd.read_parquet(edge_map_path)

    print(f"[Load] Reading link_features from {link_features_path}")
    feats = pd.read_parquet(link_features_path)

    # Basic graph-level stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print("\n=== Graph Structure ===")
    print(f"Number of nodes: {num_nodes:,}")
    print(f"Number of edges: {num_edges:,}")

    # Parse time_bin if we want to filter
    if "time_bin" in feats.columns:
        feats["time_bin"] = pd.to_datetime(feats["time_bin"])
    else:
        print("WARNING: link_features has no time_bin column; skipping time filtering.")

    # Optional time filtering
    if args.time_start is not None:
        ts = pd.to_datetime(args.time_start)
        feats = feats[feats["time_bin"] >= ts]
        print(f"[Filter] time_bin >= {ts} -> {len(feats):,} rows")
    if args.time_end is not None:
        te = pd.to_datetime(args.time_end)
        feats = feats[feats["time_bin"] < te]
        print(f"[Filter] time_bin < {te} -> {len(feats):,} rows")

    # Drop obviously bad or NaN congestion_factor
    if "congestion_factor" in feats.columns:
        feats = feats.replace([np.inf, -np.inf], np.nan)

    print("\n=== Feature Stats (Global over selected time range) ===")

    for col in ["congestion_factor", "mean_speed_kph", "mean_travel_time_s", "count"]:
        if col in feats.columns:
            series = feats[col].dropna()
            if len(series) == 0:
                print(f"{col}: no valid data")
                continue
            print(f"\n{col}:")
            print(f"  mean   = {series.mean():.4f}")
            print(f"  std    = {series.std():.4f}")
            print(f"  min    = {series.min():.4f}")
            print(f"  25pct  = {series.quantile(0.25):.4f}")
            print(f"  median = {series.median():.4f}")
            print(f"  75pct  = {series.quantile(0.75):.4f}")
            print(f"  max    = {series.max():.4f}")
        else:
            print(f"{col}: (column not in link_features)")

    # ------------------------------------------------------------------
    # Histogram of congestion_factor
    # ------------------------------------------------------------------
    if "congestion_factor" in feats.columns:
        cf = feats["congestion_factor"].dropna()
        if len(cf) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(cf, bins=50)
            plt.xlabel("Congestion factor (free-flow speed / observed speed)")
            plt.ylabel("Frequency")
            plt.title("Histogram of Congestion Factor")
            plt.grid(True, alpha=0.3)
            hist_path = f"{args.output_prefix}_congestion_hist.png"
            plt.tight_layout()
            plt.savefig(hist_path, dpi=150)
            plt.close()
            print(f"\n[Save] Congestion-factor histogram -> {hist_path}")
        else:
            print("\n[Skip] No valid congestion_factor values for histogram.")
    else:
        print("\n[Skip] No congestion_factor column for histogram.")

    # ------------------------------------------------------------------
    # Heatmap: color edges by chosen metric (averaged over time)
    # ------------------------------------------------------------------
    metric = args.metric
    if metric not in feats.columns:
        print(f"\n[Heatmap] Metric '{metric}' not found in link_features; skipping heatmap.")
        return
    
    # Aggregate metric over time by edge_ix (mean)
    metric_by_edge = (
        feats.groupby("edge_ix")[metric]
        .mean()
        .rename(f"{metric}_mean")
        .reset_index()
    )

    # Merge with edge_map to get (u, v, key)
    edge_map_metric = edge_map.merge(metric_by_edge, on="edge_ix", how="left")

    # Build (u, v, key) -> metric dict for fast lookup
    # If your edge_map has no 'key' column (you dropped it), adapt accordingly.
    breakpoint()
    has_key = "key" in edge_map_metric.columns
    if has_key:
        uvk_to_val = {
            (int(row.u), int(row.v), int(row.key)): getattr(row, f"{metric}_mean")
            for row in edge_map_metric.itertuples(index=False)
        }
    else:
        uvk_to_val = {
            (int(row.u), int(row.v)): getattr(row, f"{metric}_mean")
            for row in edge_map_metric.itertuples(index=False)
            for row in edge_map_metric.itertuples(index=False)
        }

    # Build list of metric values in the same order as G.edges(...)
    edge_values = []
    for e in G.edges(keys=True, data=True):
        u, v, k, data = e
        if has_key:
            val = uvk_to_val.get((u, v, k), np.nan)
        else:
            val = uvk_to_val.get((u, v), np.nan)
        edge_values.append(val)

    edge_values_arr = np.array(edge_values, dtype=float)

    # Handle case where many edges have NaN metric
    # We'll mask NaNs by setting them to the min value for colormap, but keep track.
    finite_vals = edge_values_arr[np.isfinite(edge_values_arr)]
    if finite_vals.size == 0:
        print(f"\n[Heatmap] No finite values for metric '{metric}'; skipping heatmap.")
        return

    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])  # robust range
    # Clip to avoid over-saturation, and fill NaNs with vmin for plotting
    vals_for_plot = np.clip(
        np.where(np.isfinite(edge_values_arr), edge_values_arr, vmin),
        vmin,
        vmax,
    )

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap("viridis")

    edge_colors = [mcolors.to_hex(cmap(norm(v))) for v in vals_for_plot]

    print(
        f"\n[Heatmap] Plotting metric '{metric}' "
        f"(aggregated mean per edge over selected time range)"
    )

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=0.8,
        bgcolor="white",
        show=False,
        close=False,
    )

    # Add a colorbar manually
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(metric)

    heatmap_path = f"{args.output_prefix}_heatmap_{metric}.png"
    plt.title(f"NYC Road Network Heatmap ({metric})")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    print(f"[Save] Heatmap -> {heatmap_path}")

if __name__ == "__main__":
    main()
