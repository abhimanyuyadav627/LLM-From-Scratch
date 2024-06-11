import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
    
    def forward(self, x):
        return x