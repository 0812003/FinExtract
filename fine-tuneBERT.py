# from transformers import Trainer, TrainingArguments
# from pretrainedBERT import model, tokenizer
# from preparedata import dataset

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./bert_invoice_classifier",
#     evaluation_strategy="no",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     # eval_dataset=eval_data
# )

# # Start training
# trainer.train()

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm
import pandas as pd
from pretrainedBERT import model, tokenizer
from preparedata import dataset
from label import df

# Assuming `df` is your dataframe containing labeled data
# Let's split it into training and evaluation sets (80/20 split)

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize Tokenizer and Model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Custom Dataset Class for Loading Data
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
        return {**{key: value.squeeze(0) for key, value in encoding.items()}, "labels": torch.tensor(label, dtype=torch.long)}

# Prepare DataLoaders for Training and Evaluation
train_dataset = InvoiceDataset(train_df, tokenizer)
eval_dataset = InvoiceDataset(eval_df, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# Training Loop Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
epochs = 5  # You can adjust the number of epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss
        running_loss += loss.item()

    # Print epoch loss
    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_dataloader)}")

    # Evaluate the model after each epoch
    model.eval()  # Set the model to evaluation mode
    eval_loss = 0.0
    eval_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch + 1}/{epochs}"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Track the loss
            eval_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            eval_accuracy += (preds == labels).sum().item()

    # Print evaluation results
    avg_eval_loss = eval_loss / len(eval_dataloader)
    avg_eval_accuracy = eval_accuracy / len(eval_df)
    print(f"Epoch {epoch + 1} - Eval Loss: {avg_eval_loss}")
    print(f"Epoch {epoch + 1} - Eval Accuracy: {avg_eval_accuracy}")

# Save Model Weights
save_path = "bert_weights.pth"
torch.save(model.state_dict(), save_path)
print(f"BERT weights saved to {save_path}")

# Load Model Weights
model.load_state_dict(torch.load(save_path))
print("BERT weights loaded successfully!")

# Save the trained model
model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")



