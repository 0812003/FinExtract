import torch
from pretrainedBERT import model, tokenizer
from sklearn.metrics import classification_report

# Define label mappings based on the trained model (num_labels=6)
LABELS = ["Vendor", "Invoice", "Other", "Horizontal Line", "Column Headers", "Data"]

def classify_text(text):
    """Classify a new text line using fine-tuned BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # Ensure index is within bounds
    if prediction < len(LABELS):
        return LABELS[prediction]
    else:
        return "Unknown"

# Example usage
print(classify_text("Invoice 12345 dated 01/02/2024"))
print(classify_text("Vendor: ABC Corporation"))

def evaluate_model(dataset):
    """Evaluate BERT model on labeled test dataset."""
    predictions, actuals = [], []
    
    for item in dataset:
        # Convert input tensor back to text
        text = tokenizer.decode(item["input_ids"].squeeze(), skip_special_tokens=True)
        label = item["labels"].item()

        pred_label = classify_text(text)
        pred_index = LABELS.index(pred_label)  # Convert label to integer index

        predictions.append(pred_index)
        actuals.append(label)

    print(classification_report(actuals, predictions, target_names=LABELS))

# import pdfplumber
# import os
# import pandas as pd
# import torch
# from pretrainedBERT import model, tokenizer
# from sklearn.metrics import classification_report

# # Label mappings (should match model's num_labels=6)
# LABELS = ["Vendor", "Invoice", "Other", "Horizontal Line", "Column Headers", "Data"]

# def extract_text_from_pdfs(pdf_path):
#     """Extract text from a single PDF file."""
#     extracted_data = []  # Store text and expected labels
#     if not pdf_path.endswith(".pdf"):
#         print(f"Error: Expected a PDF file, but got {pdf_path}")
#         return pd.DataFrame(extracted_data)

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 for line in text.split("\n"):
#                     extracted_data.append({"text": line.strip()})
    
#     return pd.DataFrame(extracted_data)

# # Path to your single PDF file
# pdf_path = r"F:\HACK-SPHERE\Model\data\Sample (1).pdf"

# # Extract text from the PDF
# df_extracted = extract_text_from_pdfs(pdf_path)
# print(df_extracted.head())  # Check sample output
# def classify_text(text):
#     """Classify a new text line using fine-tuned BERT."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()  # Ensure model is in evaluation mode

#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     inputs = {key: val.to(device) for key, val in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)

#     prediction = torch.argmax(outputs.logits, dim=-1).item()
    
#     return LABELS[prediction] if prediction < len(LABELS) else "Unknown"

# # Apply classification to extracted text
# if not df_extracted.empty:
#     df_extracted["predicted_label"] = df_extracted["text"].apply(classify_text)
#     print(df_extracted.head(20))  # Check sample output
# else:
#     print("No text extracted from the PDF.")

# # Evaluation (if actual labels exist)
# if "actual_label" in df_extracted.columns:
#     df_extracted["actual_label_index"] = df_extracted["actual_label"].apply(lambda x: LABELS.index(x))
#     df_extracted["predicted_label_index"] = df_extracted["predicted_label"].apply(lambda x: LABELS.index(x))

#     # Generate Classification Report
#     print(classification_report(df_extracted["actual_label_index"], df_extracted["predicted_label_index"], target_names=LABELS))
# else:
#     print("No actual labels found. Please label a small subset manually for evaluation.")

