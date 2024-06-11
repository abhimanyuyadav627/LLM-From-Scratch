import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, input_text, tokenizer,max_length = 256, stride = 1) -> None:
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        encoded_text = self.tokenizer.encode(input_text)
        for i in range(0, len(encoded_text) - max_length, stride):
            self.input_ids.append(torch.tensor(encoded_text[i:i + max_length]))
            self.target_ids.append(torch.tensor(encoded_text[i+1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, max_length = 256, stride = 1, batch_size = 4, shuffle = False):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def input_embedding_pipeline(txt, max_length = 256, stride = 1, batch_size = 4, shuffle = False):
    pass
#REMINDER: Need to implement Embedding Layer & Positional encodings.

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader(raw_text, max_length=4, stride=1, batch_size=2 , shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)