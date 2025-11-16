from pyogrio import read_dataframe
import geopandas as gpd
import pandas as pd

GDB = "lion/lion.gdb"  # adjust if your path differs

# --- load layers using your actual names ---
nodes_gdf = read_dataframe(GDB, layer="node").to_crs(4326)
edges_gdf = read_dataframe(GDB, layer="lion").to_crs(4326)

# helper: pick a column by trying several common names (case-insensitive)
def pick(cols, candidates):
    up = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in up:
            return up[cand.lower()]
    raise KeyError(f"Need one of {candidates}, have: {list(cols)}")

# Try to find IDs (works across LION variants)
node_id_col = pick(nodes_gdf.columns, ["NODEID", "NODE_ID", "NODE", "nodeid", "node_id", "node"])
from_col    = pick(edges_gdf.columns, ["NODEIDFROM", "FROMNODE", "FRM_NODE", "FR_NODE", "nodeidfrom", "fromnode", "frm_node", "fr_node"])
to_col      = pick(edges_gdf.columns,   ["NODEIDTO",   "TONODE",   "TO_NODE",  "to_node",    "nodeidto",   "tonode"])

# Optional attributes
trafdir_col = next((c for c in edges_gdf.columns if c.lower() in {"trafdir","oneway"}), None)
lanes_col   = next((c for c in edges_gdf.columns if c.lower() in {"lanes","lane"}), None)
speed_col   = next((c for c in edges_gdf.columns if c.lower() in {"speed_lim","maxspeed","speedlimit"}), None)
name_col    = next((c for c in edges_gdf.columns if c.lower() in {"street","name","full_stree"}), None)

# --- nodes table ---
nodes = pd.DataFrame({
    "node_id": nodes_gdf[node_id_col].astype(str),
    "lon": nodes_gdf.geometry.x,
    "lat": nodes_gdf.geometry.y,
})

# --- edges base (u,v, key, oneway) ---
edges = pd.DataFrame({
    "u": edges_gdf[from_col].astype(str),
    "v": edges_gdf[to_col].astype(str),
})
edges["key"] = 0

# oneway detection: LION trafdir often EB/WB/NB/SB for one-way, 'TWOWAY' for two-way
if trafdir_col:
    traf = edges_gdf[trafdir_col].astype(str).str.upper()
    oneway = ~traf.str.contains("TWO")   # True if EB/WB/NB/SB, False if TWO(WAY)
else:
    oneway = pd.Series(False, index=edges.index)
edges["oneway"] = oneway.values

# compute length in meters from geometry
edges_m = read_dataframe(GDB, layer="lion").to_crs(6539)  # meters
edges["length_m"] = edges_m.length.values

# attach optional attrs if present
if lanes_col: edges["lanes"] = edges_gdf[lanes_col]
if speed_col: edges["maxspeed"] = edges_gdf[speed_col]
if name_col:  edges["name"] = edges_gdf[name_col]

# edge_id (u-v-key)
edges["edge_id"] = edges["u"] + "-" + edges["v"] + "-" + edges["key"].astype(str)

# duplicate reverse for two-way
twoway = ~edges["oneway"]
edges_rev = edges.loc[twoway].copy()
edges_rev[["u","v"]] = edges_rev[["v","u"]].values
edges_rev["edge_id"] = edges_rev["u"] + "-" + edges_rev["v"] + "-" + edges_rev["key"].astype(str)

edges_full = pd.concat([edges, edges_rev], ignore_index=True)

# --- save ---
nodes.to_parquet("nodes.parquet", index=False)
edges_full.to_parquet("edges.parquet", index=False)
print("Wrote nodes.parquet and edges.parquet")
