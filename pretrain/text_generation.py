import torch
import tiktoken
from gpt_architecture.custom_gpt import GPTModel, GPT_CONFIG_124M

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
    def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

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