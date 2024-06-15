import numpy as np
import torch
from gpt_architecture.custom_gpt import GPTModel,GPT_CONFIG_124M
from pretrain.gpt_download import download_and_load_gpt2
import urllib.request


def assign(left, right):
    if left.shape != right.shape:
       raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right)) 

def load_weights_into_gpt(gpt,params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # loading weights in Transformer Block
    for block_no in range(len(params["blocks"])):
        # loading weights in Attention Module
        q_w, k_w, v_w = np.split(params["blocks"][block_no]["attn"]["c_attn"]["w"],3,axis = -1)
        
        # loading weights in query, key and value tensors
        gpt.transformer_blocks[block_no].attn.W_query.weight = assign(
        gpt.transformer_blocks[block_no].attn.W_query.weight, q_w.T  
        )
        gpt.transformer_blocks[block_no].attn.W_key.weight = assign(
        gpt.transformer_blocks[block_no].attn.W_key.weight, k_w.T  
        )
        gpt.transformer_blocks[block_no].attn.W_value.weight = assign(
        gpt.transformer_blocks[block_no].attn.W_value.weight, v_w.T  
        ) 

        q_b, k_b, v_b = np.split(params["blocks"][block_no]["attn"]["c_attn"]["b"],3,axis = -1)

        # loading biases in query, key and value tensors
        gpt.transformer_blocks[block_no].attn.W_query.bias = assign(
        gpt.transformer_blocks[block_no].attn.W_query.bias, q_b  
        )
        gpt.transformer_blocks[block_no].attn.W_key.bias = assign(
        gpt.transformer_blocks[block_no].attn.W_key.bias, k_b 
        )
        gpt.transformer_blocks[block_no].attn.W_value.bias = assign(
        gpt.transformer_blocks[block_no].attn.W_value.bias, v_b 
        )
    
        gpt.transformer_blocks[block_no].attn.out_proj.weight = assign(
            gpt.transformer_blocks[block_no].attn.out_proj.weight,
            params["blocks"][block_no]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[block_no].attn.out_proj.bias = assign(
            gpt.transformer_blocks[block_no].attn.out_proj.bias,
            params["blocks"][block_no]["attn"]["c_proj"]["b"])
        
        # loading weights in Feed Forward Module
        gpt.transformer_blocks[block_no].ff.layers[0].weight = assign(
            gpt.transformer_blocks[block_no].ff.layers[0].weight,
            params["blocks"][block_no]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[block_no].ff.layers[0].bias = assign(
            gpt.transformer_blocks[block_no].ff.layers[0].bias,
            params["blocks"][block_no]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[block_no].ff.layers[2].weight = assign(
            gpt.transformer_blocks[block_no].ff.layers[2].weight,
            params["blocks"][block_no]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[block_no].ff.layers[2].bias = assign(
            gpt.transformer_blocks[block_no].ff.layers[2].bias,
            params["blocks"][block_no]["mlp"]["c_proj"]["b"])
        
        # loading weights in LayerNormalisation modules of Transfomer Blocks

        gpt.transformer_blocks[block_no].norm1.scale = assign(
            gpt.transformer_blocks[block_no].norm1.scale,
            params["blocks"][block_no]["ln_1"]["g"])
        gpt.transformer_blocks[block_no].norm1.shift = assign(
            gpt.transformer_blocks[block_no].norm1.shift,
            params["blocks"][block_no]["ln_1"]["b"])
        gpt.transformer_blocks[block_no].norm2.scale = assign(
            gpt.transformer_blocks[block_no].norm2.scale,
            params["blocks"][block_no]["ln_2"]["g"])
        gpt.transformer_blocks[block_no].norm2.shift = assign(
            gpt.transformer_blocks[block_no].norm2.shift,
            params["blocks"][block_no]["ln_2"]["b"])
    
    # loading weights in the final LayerNormalisation layer of the network
    gpt.final_layer_norm.scale = assign(gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift = assign(gpt.final_layer_norm.shift, params["b"])
    
    # loading weights in the final Linear output layer of the network.
    # NOTE: The original GPT-2 model by OpenAI reused the token embedding weights in the output layer to reduce the total number of parameters, which is a concept known as weight tying.
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def get_gpt_with_openai_gpt2_weights():

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch05/"
        "01_main-chapter-code/gpt_download.py"
    )
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url,filename)
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length":1024, "qkv_bias":True})
    gpt = GPTModel(NEW_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_weights_into_gpt(gpt,params)
    return gpt

