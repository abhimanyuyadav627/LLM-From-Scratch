import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,configuration) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(configuration["emb_dim"],4 * configuration["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * configuration["emb_dim"], configuration["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)
    