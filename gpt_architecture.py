import torch
import torch.nn as nn 
from transformer_block import TransformerBlock
from layer_normalisation import LayerNorm

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, configuration) -> None:
        self.tok_emb = nn.Embedding(configuration["vocab_size"],configuration["emb_dim"])
        self.pos_emb = nn.Embedding(configuration["context_length"], configuration["emb_dim"])
        self.dropout_layer_embedding = nn.Dropout(configuration["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(configuration) for _ in configuration["n_layers"]])
        self.final_layer_norm = LayerNorm(configuration["emb_dim"])
        self.out_head = nn.Linear(configuration["emb_dim"],configuration["vocab_size"], bias = False)
    
    def forward(self, input_idx):
        #input_idx : (batch,num_tokens)
        batch_size, sequence_length = input_idx.shape

        token_embedding = self.tok_emb(input_idx)
        pos_embedding = self.pos_emb(torch.arange(sequence_length, device= input_idx.device))

        x = token_embedding + pos_embedding
        x = self.dropout_layer_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits