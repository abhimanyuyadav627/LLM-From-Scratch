import torch
import tiktoken
import torch.nn as nn 
from gpt_architecture.modules.transformer_block import TransformerBlock
from gpt_architecture.modules.layer_normalisation import LayerNorm

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 256, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, configuration) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(configuration["vocab_size"],configuration["emb_dim"])
        self.pos_emb = nn.Embedding(configuration["context_length"], configuration["emb_dim"])
        self.dropout_layer_embedding = nn.Dropout(configuration["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(configuration) for _ in range(configuration["n_layers"])])
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
    

if __name__ == "__main__":
    #Sanity check code
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")