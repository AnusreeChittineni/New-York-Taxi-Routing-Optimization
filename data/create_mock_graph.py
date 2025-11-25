import torch
from torch_geometric.data import Data

def create_mock_graph(path):
    num_nodes = 50
    num_edges = 200
    
    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random edge attributes (length, capacity, etc.)
    # Assuming 4 features as per typical setup
    edge_attr = torch.rand(num_edges, 4)
    
    # Random node features
    x = torch.rand(num_nodes, 2)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes
    
    torch.save(data, path)
    print(f"Mock graph saved to {path}")

if __name__ == "__main__":
    create_mock_graph("data/mock_graph.pt")
