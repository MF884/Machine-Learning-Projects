import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    RagTokenizer, RagRetriever, RagSequenceForGeneration
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spacy
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

# Optional if spaCy isn't needed anymore
nlp = spacy.load("en_core_web_sm")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_LEN = 256
BATCH_SIZE = 16 if torch.cuda.is_available() else 8
EPOCHS = 10
LEARNING_RATE = 1e-5
THRESHOLD = 0.5

# Enable tqdm for pandas
tqdm.pandas()

class DataProcessor:
    def __init__(self):
        self.sentiment_keywords = {
            'positive': {'thanks': 2, 'great': 2, 'solved': 3, 'love': 2, '😊': 3},
            'negative': {'😡': 3, 'sucks': 3, 'hate': 3, 'broken': 2, 'painful': 2}
        }
        self.issue_keywords = {
            'technical': ['broken', 'crash', 'freezes', 'bug', 'error'],
            'service': ['delay', 'refund', 'callback', 'support'],
            'performance': ['battery', 'slow', 'speed', 'drains']
        }

    def load_and_validate_data(self, file_path):
        df = pd.read_csv(file_path)
        df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
        df = df.dropna(subset=['text'])
        df = df.sample(n=5000)  # Sample before label generation for speed
        df['labels'] = df['text'].progress_apply(self.generate_labels)
        assert len(df['text']) == len(df['labels']), "Text and labels length mismatch"
        return df

    def generate_labels(self, text):
        text_lower = text.lower()
        labels = []
        pos_score = sum(weight for word, weight in self.sentiment_keywords['positive'].items() if word in text_lower)
        neg_score = sum(weight for word, weight in self.sentiment_keywords['negative'].items() if word in text_lower)

        if pos_score > neg_score:
            labels.append('positive')
        elif neg_score > pos_score:
            labels.append('negative')
        else:
            labels.append('neutral')

        for issue_type, keywords in self.issue_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                labels.append(issue_type)

        return labels

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def safe_train_test_split(texts, labels, test_size=0.2):
    return train_test_split(texts, labels, test_size=test_size, random_state=42, stratify=None)

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()

def main():
    processor = DataProcessor()
    df = processor.load_and_validate_data('twcs.csv')

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['labels'])
    X_train, X_test, y_train, y_test = safe_train_test_split(df['text'], y)
    X_train, X_val, y_train, y_val = safe_train_test_split(X_train, y_train, test_size=0.25)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = TweetDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = TweetDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = TweetDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(mlb.classes_)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, inputs['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask', 'labels']}
                outputs = model(**inputs)
                val_loss += loss_fn(outputs.logits, inputs['labels']).item()
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    plot_loss(train_losses, val_losses)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
            labels = batch['labels'].numpy()
            outputs = model(**inputs)
            preds = (torch.sigmoid(outputs.logits) > THRESHOLD).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=mlb.classes_, zero_division=0))

  # === RAG ===
    print("\nGenerating a response using RAG:")
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    input_text = "Why does my phone keep restarting?"
    inputs = rag_tokenizer([input_text], return_tensors="pt")
    # Remove token_type_ids from the inputs dictionary
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"] 
    generated = rag_model.generate(**inputs)
    print("RAG Output:", rag_tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
if __name__ == "__main__":
    main()
