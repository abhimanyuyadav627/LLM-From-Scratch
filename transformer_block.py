import torch
import torch.nn as nn
from self_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_normalisation import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, configuration) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = configuration["emd_dim"],
            d_out = configuration["emd_dim"],
            context_length = configuration["context_length"],
            num_heads = configuration["n_heads"],
            dropout = configuration["drop_rate"],
            qkv_bias = configuration["qkv_bias"]
        )
        self.ff = FeedForward(configuration)
        self.norm1 = LayerNorm(configuration["emd_dim"])
        self.norm2 = LayerNorm(configuration["emd_dim"])
        self.drop_shortcut = nn.Dropout(configuration["drop_rate"])

    def forward(self, x):

        skip_connection = x

        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + skip_connection

        skip_connection = x
        
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + skip_connection
        return x