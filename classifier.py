import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


labels = ["product", "terms", "store_info", "else"]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_texts(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    del tokens["token_type_ids"]  # Bỏ `token_type_ids` để tránh lỗi
    return tokens

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.fc = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
        attn_output, _ = self.attention(cls_embedding, cls_embedding, cls_embedding)
        output = self.fc(self.dropout(attn_output.squeeze(0)))
        return output
    

def load_model(filename="training/model.pkl"):
    print("Loading model")
    with open(filename, "rb") as f:
        return pickle.load(f)
    
# Hàm dự đoán
def predict(text, model):
    model.eval()
    inputs = tokenize_texts([text])
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs, dim=1).item()
    return label_encoder.inverse_transform([predicted_label])[0]