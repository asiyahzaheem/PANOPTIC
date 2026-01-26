"""from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    ""
    ResNet18 backbone -> 512-d embedding per patch + classifier head.
    Future GNN: use embeddings as node features.
    ""
    def __init__(self, pretrained: bool = True, emb_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # avgpool output
        self.emb_dim = emb_dim
        self.classifier = nn.Linear(emb_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)     # [B,512,1,1]
        f = f.flatten(1)         # [B,512]
        return f

    def forward(self, x: torch.Tensor):
        f = self.extract_features(x)
        logit = self.classifier(f)
        return logit, f
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Embedder(nn.Module):
    """
    Pure embedding extractor.
    Input:  [N, 3, 224, 224]
    Output: [N, 512]
    """
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super().__init__()
        base = resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        self.out_dim = 512

    def forward(self, x):
        f = self.features(x)            # [N,512,1,1]
        f = f.flatten(1)                # [N,512]
        return f


class LinearHead(nn.Module):
    """
    Small head to map embeddings -> logit (binary).
    """
    def __init__(self, in_dim=512):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z):
        return self.fc(z)               # [N,1]
