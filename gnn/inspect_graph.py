import torch
data = torch.load('data/nyc_graph.pt', weights_only=False)
print(data)
print(data.keys)
if hasattr(data, 'edge_attr'):
    print(f"Edge attr shape: {data.edge_attr.shape}")
# Check if there's any attribute that looks like segment IDs
for key, item in data:
    if key == 'edge_index': continue
    if torch.is_tensor(item) and item.size(0) == data.edge_index.size(1):
        print(f"Edge attribute: {key}, shape: {item.shape}, dtype: {item.dtype}")
        if item.dtype in [torch.int, torch.long]:
             print(f"Sample values: {item[:5]}")
