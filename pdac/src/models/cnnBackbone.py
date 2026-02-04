"""
ResNet18-based feature extractor for CT scan embeddings
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Pretrained ResNet18 with the final classification layer removed to extract 512-d embeddings
class ResNet18Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        # load pretrained ResNet18 and strip off the final FC layer
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract features and flatten to get embedding vector
        f = self.features(x)
        f = f.flatten(1)
        return f

