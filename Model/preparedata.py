# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer
# import torch
# from label import df
# from pretrainedBERT import model, tokenizer
# class InvoiceDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_length=128):
#         self.texts = dataframe["text"].tolist()
#         self.labels = dataframe["label"].tolist()
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text, 
#             padding="max_length", 
#             truncation=True, 
#             max_length=self.max_length, 
#             return_tensors="pt"
#         )
#         return {**encoding, "labels": torch.tensor(label, dtype=torch.long)}

# # Load dataset
# dataset = InvoiceDataset(df, tokenizer)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
from label import df  # Make sure your dataframe df has the correct format
from pretrainedBERT import model, tokenizer  # Ensure you have loaded model and tokenizer

# Training Loop Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class InvoiceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        # Flatten the encoding tensors and return the data
        return {**{key: value.squeeze(0) for key, value in encoding.items()}, "labels": torch.tensor(label, dtype=torch.long)}

# Load dataset
dataset = InvoiceDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Example of iterating over the DataLoader
for batch in dataloader:
    print(batch)
    break  # Just printing the first batch
