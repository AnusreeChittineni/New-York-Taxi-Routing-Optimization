# save as build_graph_from_json.py  (run: python build_graph_from_json.py)
import json, gzip, sys
import networkx as nx

def stream_json_lines(path):
    # supports .gz or plain
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_graph(nodes_path="nodes.ndjson", edges_path="edges.ndjson"):
    G = nx.MultiDiGraph(name="NYC")

    # 1) Add nodes
    for rec in stream_json_lines(nodes_path):
        G.add_node(rec["node_id"], lon=rec["lon"], lat=rec["lat"])

    # 2) Add edges (expand two-way into both directions)
    cnt = 0
    for rec in stream_json_lines(edges_path):
        u = rec["u"]; v = rec["v"]
        data = {
            "length_m": rec.get("length_m"),
            "name": rec.get("name"),
            "oneway": bool(rec.get("oneway", False)),
        }
        G.add_edge(u, v, **data)
        cnt += 1
        if not data["oneway"]:
            G.add_edge(v, u, **data)
            cnt += 1

    print(f"Nodes: {G.number_of_nodes()}  Directed edges (after expand): {G.number_of_edges()}")
    return G

if __name__ == "__main__":
    nodes = sys.argv[1] if len(sys.argv)>1 else "nodes.ndjson"
    edges = sys.argv[2] if len(sys.argv)>2 else "edges.ndjson"
    G = build_graph(nodes, edges)

    # optional: persist a light adjacency JSON for quick use
    adj = {u: [v for v,_k in G[u].items()] for u in G.nodes()}
    with open("adjacency.json", "w") as f:
        json.dump(adj, f)
    print("Wrote adjacency.json")
