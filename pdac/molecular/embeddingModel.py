"""
Simple linear netowrk that compresses molecular feature vectors into a smaller embedding
"""

import torch.nn as nn

class MolecularEmbedder(nn.Module):
    def __init__(self, input_dim, emb_dim=256, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

