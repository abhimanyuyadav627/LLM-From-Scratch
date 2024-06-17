import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import tiktoken
from torch.utils.data import DataLoader



class SpamDataset(Dataset):
    
    def __init__(self, df, tokenizer, max_length = None, pad_token_id=50256):
        self.data = df

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length == None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) 
                              for encoded_text in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        
        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
        
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            text_length = len(encoded_text)
            if text_length > max_length:
                max_length = text_length
        return max_length

# -----------------------------------------------------------------------------------------------------
# Code to download the data for finetuning
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "../data/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

if not data_file_path.exists():
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
# -----------------------------------------------------------------------------------------------------


def preprocess_and_create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    return balanced_df



# -----------------------------------------------------------------------------------------------------
# Code to prepare train, validation and test datasets.
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) #suffling the entire dataframe
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

# -----------------------------------------------------------------------------------------------------

def get_data_loaders(tokenizer):
    
    balanced_df = preprocess_and_create_balanced_dataset(pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"]))
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    
    train_dataset = SpamDataset(train_df,max_length=None,tokenizer=tokenizer)
    val_dataset = SpamDataset(validation_df,max_length=train_dataset.max_length,tokenizer=tokenizer)
    test_dataset = SpamDataset(test_df,max_length=train_dataset.max_length,tokenizer=tokenizer)
    
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True,)
    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader,val_loader,test_loader = get_data_loaders(tokenizer)

    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
