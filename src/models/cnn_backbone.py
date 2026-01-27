from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(m.children())[:-1])  # up to avgpool
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        f = self.features(x)          # [B,512,1,1]
        f = f.flatten(1)              # [B,512]
        return f
