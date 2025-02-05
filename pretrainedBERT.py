from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)  # 6 classes
# Labeling Logic
# Line Type	Label
# Vendor Name (First Line)	0
# Invoice Line	1
# Other Text	2
# Horizontal Line (-----)	3
# Column Headers	4
# Data	8