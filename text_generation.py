import torch
import tiktoken
from gpt_architecture import GPTModel, GPT_CONFIG_124M

class TextGenerator:

    def generate_text_simple(self,model,idx,max_new_tokens,context_size):
        
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
    

if __name__ == "__main__":
    #Sanity check code
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch = torch.stack(batch, dim=0)
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    # model.eval() .eval() mode, which disables random components like dropout, which are only used during training
    out = TextGenerator().generate_text_simple(
        model=model,
        idx=batch,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"])
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"Generated_Text: {decoded_text}")