import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, configuration) -> None:
        super().__init__()
    
    def forward(self, x):
        return x