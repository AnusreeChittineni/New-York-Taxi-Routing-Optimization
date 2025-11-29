import torch
import pandas as pd
import numpy as np

data = torch.load('data/nyc_graph.pt', weights_only=False)
graph_ids = set(data.edge_osm_id.numpy())

df = pd.read_csv('data/nyc_road_safety_analysis.csv')
csv_ids = set(df['SegmentID'].unique())

intersection = graph_ids.intersection(csv_ids)
print(f"Graph IDs: {len(graph_ids)}")
print(f"CSV IDs: {len(csv_ids)}")
print(f"Intersection: {len(intersection)}")

if intersection:
    print(f"Sample intersection: {list(intersection)[:5]}")
else:
    print("No intersection found.")
    print(f"Sample Graph IDs: {list(graph_ids)[:5]}")
    print(f"Sample CSV IDs: {list(csv_ids)[:5]}")
