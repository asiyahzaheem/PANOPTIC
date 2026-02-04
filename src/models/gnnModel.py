"""
GraphSAGE-based GNN classifier for pancreatic cancer subtype prediction
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# try to import torch_geometric (required for GNN layers)
try:
    from torch_geometric.nn import SAGEConv
except Exception as e:
    SAGEConv = None

# Two-layer GraphSAGE network with dropout for node classification
class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGEClassifier.")

        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # first graph conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # second graph conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # final linear layer for classification
        return self.lin(x)
