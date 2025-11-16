# save as export_to_json.py  (run: python export_to_json.py)
from pyogrio import read_dataframe
import pandas as pd

GDB = "lion/lion.gdb"

# 1) Read layers you discovered
nodes_gdf = read_dataframe(GDB, layer="node").to_crs(4326)
edges_gdf = read_dataframe(GDB, layer="lion").to_crs(4326)
edges_gdf_m = read_dataframe(GDB, layer="lion").to_crs(6539)  # meters for length

# — pick columns robustly —
def pick(cols, cands):
    up = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in up: return up[c.lower()]
    raise KeyError(f"Need one of {cands}; have {list(cols)}")

node_id = pick(nodes_gdf.columns, ["NODEID","NODE_ID","NODE"])
from_id = pick(edges_gdf.columns, ["NODEIDFROM","FROMNODE","FRM_NODE","FR_NODE"])
to_id   = pick(edges_gdf.columns, ["NODEIDTO","TONODE","TO_NODE"])
trafdir = next((c for c in edges_gdf.columns if c.lower() in {"trafdir","oneway"}), None)
namecol = next((c for c in edges_gdf.columns if c.lower() in {"street","name","full_stree"}), None)

# 2) Nodes NDJSON: node_id, lon, lat
nodes_df = pd.DataFrame({
    "node_id": nodes_gdf[node_id].astype(str),
    "lon": nodes_gdf.geometry.x,
    "lat": nodes_gdf.geometry.y,
})
nodes_df.to_json("nodes.ndjson", orient="records", lines=True)

# 3) Edges NDJSON (undirected in file; we’ll “expand” when loading)
oneway_series = None
if trafdir:
    t = edges_gdf[trafdir].astype(str).str.upper()
    oneway_series = ~t.str.contains("TWO")
else:
    oneway_series = pd.Series(False, index=edges_gdf.index)

edges_df = pd.DataFrame({
    "u": edges_gdf[from_id].astype(str),
    "v": edges_gdf[to_id].astype(str),
    "name": edges_gdf[namecol] if namecol else None,
    "oneway": oneway_series,
    "length_m": edges_gdf_m.length.round(3),
})
edges_df.to_json("edges.ndjson", orient="records", lines=True)

print("Wrote nodes.ndjson and edges.ndjson")
