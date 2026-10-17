# Install transformers if not already
!pip install transformers torch --quiet

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Example text
text = "BERT models are great for NLP tasks!"
inputs = tokenizer(text, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# [CLS] token embedding (sentence-level representation)
cls_embedding = outputs.last_hidden_state[:, 0, :]

print("CLS Embedding shape:", cls_embedding.shape)
