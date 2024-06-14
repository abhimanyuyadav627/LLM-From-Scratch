import torch
import tiktoken
from gpt_architecture import GPTModel, GPT_CONFIG_124M

class TextGenerator:
    @staticmethod
    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        # .unsqueeze(0) adds the batch dimension
        return encoded_tensor
    @staticmethod
    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0) # Remove batch dimension
        return tokenizer.decode(flat.tolist())
    @staticmethod
    def generate_text_simple(model,idx,max_new_tokens,context_size):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:,-1,:]
            # softmax function is monotonic, meaning it preserves the order of its inputs
            # when transformed into outputs. So, in practice, the softmax step is redundant since the position
            # with the highest score in the softmax output tensor is the same position in the logit tensor.
            probas = torch.softmax(logits,dim = -1)
            # greedy decoding
            idx_next = torch.argmax(probas, dim = -1, keepdim = True)
            idx = torch.cat((idx,idx_next), dim = 1)
        return idx
    
    
    @staticmethod
    def generate_text(model,idx,max_new_tokens,context_size,temperature=0.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:,-1,:]
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val,torch.tensor(float('-inf')).to(logits.device),logits)
            
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # if idx_next == eos_id: #condition for termination is eos token is generated.
            #     break
            idx = torch.cat((idx,idx_next), dim = 1)
        return idx

if __name__ == "__main__":
    #Sanity check code
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every day holds a"
    batch = TextGenerator.text_to_token_ids(txt1,tokenizer)
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    # model.eval() .eval() mode, which disables random components like dropout, which are only used during training
    out = TextGenerator.generate_text_simple(
        model=model,
        idx=batch,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"])
    decoded_text = TextGenerator.token_ids_to_text(out,tokenizer)
    print(f"Generated_Text: {decoded_text}")