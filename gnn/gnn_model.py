"""GNN architecture and persistence utilities for travel-time prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


def detect_device(preferred: Optional[str] = None) -> torch.device:
    """Pick CUDA, MPS, or CPU in that order."""

    if preferred:
        preferred = preferred.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TravelTimeGNN(nn.Module):
    """Simple GraphSAGE stack for per-edge travel time prediction."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1, num_layers: int = 3):
        super().__init__()
        layers = []
        input_dim = in_channels
        for _ in range(num_layers - 1):
            layers.append(SAGEConv(input_dim, hidden_channels))
            input_dim = hidden_channels
        layers.append(SAGEConv(input_dim, hidden_channels))
        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=0.15)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 0, device=x.device)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
        src, dst = edge_index
        edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=-1) if edge_attr.numel() else torch.cat(
            [x[src], x[dst]], dim=-1
        )
        return self.edge_mlp(edge_feat).squeeze(-1)


def build_model(in_channels: int, hidden_channels: int, out_channels: int = 1) -> TravelTimeGNN:
    """Factory for TravelTimeGNN."""

    return TravelTimeGNN(in_channels, hidden_channels, out_channels)


def save_model(model: nn.Module, path: str | Path) -> None:
    """Persist model weights."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, in_channels: int, hidden_channels: int, out_channels: int = 1) -> TravelTimeGNN:
    """Load model weights into a new instance."""

    model = build_model(in_channels, hidden_channels, out_channels)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model
