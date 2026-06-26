import os
import time
import yaml
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import mlflow
import mlflow.sklearn
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Text Preprocessing Utilities for LSTM ---
def simple_tokenize(text: str) -> List[str]:
    """Tokenizes text by lowercasing and splitting on whitespace and punctuation."""
    import re
    text = text.lower().replace("<br />", " ")
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

class Vocab:
    def __init__(self, max_vocab_size: int = 20000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.vocab_size = 2

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        
        most_common = counter.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        logger.info(f"Vocabulary built with size {self.vocab_size}")

    def numericalize(self, text: str, max_length: int) -> List[int]:
        tokens = simple_tokenize(text)
        indices = [self.word2idx.get(token, 1) for token in tokens[:max_length]]
        # Padding
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        return indices

class IMDbBaselineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_length: int):
        self.labels = df["label"].values
        self.texts = df["text"].values
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        numericalized = self.vocab.numericalize(self.texts[idx], self.max_length)
        return torch.tensor(numericalized, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# --- LSTM Model Definition ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout if n_layers > 1 else 0.0, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            # Concat the final forward and backward hidden states
            hidden_cat = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden_cat = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden_cat)

# --- Training Helper Functions ---
def train_logistic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, experiment_name: str) -> Dict[str, Any]:
    logger.info("Training Logistic Regression baseline...")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Logistic_Regression"):
        start_time = time.time()
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        X_train = vectorizer.fit_transform(train_df["text"])
        X_test = vectorizer.transform(test_df["text"])
        y_train = train_df["label"].values
        y_test = test_df["label"].values
        
        # Fit model
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Inference speed test
        start_inf = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = (time.time() - start_inf) / len(test_df)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log to MLflow
        mlflow.log_param("model_type", "Logistic_Regression")
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.log_metric("inference_time_per_sample_sec", inference_time)
        
        # Save vectorizer and model as artifacts
        os.makedirs("models", exist_ok=True)
        lr_bundle = {"vectorizer": vectorizer, "model": model}
        with open("models/logistic_regression_bundle.pkl", "wb") as f:
            pickle.dump(lr_bundle, f)
            
        mlflow.log_artifact("models/logistic_regression_bundle.pkl")
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"Logistic Regression: Accuracy={acc:.4f}, F1={f1:.4f}, Train Time={training_time:.2f}s")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "training_time": training_time, "inference_time": inference_time}

def train_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame, experiment_name: str, 
               epochs: int = 5, batch_size: int = 64, lr: float = 1e-3, 
               device_str: str = "auto") -> Dict[str, Any]:
    logger.info("Training LSTM baseline...")
    mlflow.set_experiment(experiment_name)
    
    # Set device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Vocab & Dataset creation
    vocab = Vocab(max_vocab_size=20000)
    vocab.build_vocab(train_df["text"].tolist())
    
    max_length = 128
    train_dataset = IMDbBaselineDataset(train_df, vocab, max_length)
    test_dataset = IMDbBaselineDataset(test_df, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model definition
    model = LSTMClassifier(
        vocab_size=vocab.vocab_size, 
        embedding_dim=100, 
        hidden_dim=128, 
        output_dim=2, 
        n_layers=2, 
        bidirectional=True, 
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    with mlflow.start_run(run_name="LSTM_Classifier"):
        # Log parameters
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("vocab_size", vocab.vocab_size)
        mlflow.log_param("embedding_dim", 100)
        mlflow.log_param("hidden_dim", 128)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("bidirectional", True)
        mlflow.log_param("dropout", 0.3)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * texts.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            epoch_loss /= len(train_loader.dataset)
            epoch_acc = correct / total
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        start_inf = time.time()
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(device)
                outputs = model(texts)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        inference_time = (time.time() - start_inf) / len(test_df)
        
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
        auc = roc_auc_score(all_labels, all_probs)
        
        # Log test metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.log_metric("inference_time_per_sample_sec", inference_time)
        
        # Save model and vocab
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/lstm_model.pt")
        with open("models/lstm_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
            
        mlflow.log_artifact("models/lstm_model.pt")
        mlflow.log_artifact("models/lstm_vocab.pkl")
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        
        logger.info(f"LSTM: Accuracy={acc:.4f}, F1={f1:.4f}, Train Time={training_time:.2f}s")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "training_time": training_time, "inference_time": inference_time}

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Train Baseline Models for IMDb Sentiment Analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--quick-test", action="store_true", help="Subset datasets for rapid testing")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Configure MLflow SQLite tracking
    tracking_uri = config["mlflow"].get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config["mlflow"].get("experiment_name", "IMDb-Sentiment-Analysis")
    
    # Load dataset
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    
    logger.info(f"Loading datasets from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # If quick-test or configured subset sizes
    train_size = 500 if args.quick_test else config["data"].get("train_sample_size")
    val_size = 100 if args.quick_test else config["data"].get("val_sample_size")
    
    if train_size is not None:
        train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=42).reset_index(drop=True)
        logger.info(f"Quick Test / Config: Train subset selected with size {len(train_df)}")
    if val_size is not None:
        test_df = test_df.sample(n=min(val_size, len(test_df)), random_state=42).reset_index(drop=True)
        logger.info(f"Quick Test / Config: Test subset selected with size {len(test_df)}")
        
    # Train Logistic Regression
    lr_metrics = train_logistic_regression(train_df, test_df, experiment_name)
    
    # Train LSTM
    lstm_epochs = 3 if args.quick_test else 5
    lstm_metrics = train_lstm(
        train_df, 
        test_df, 
        experiment_name, 
        epochs=lstm_epochs, 
        batch_size=32 if args.quick_test else 64, 
        device_str=config["training"].get("device", "auto")
    )
    
    logger.info("Baseline training completed successfully.")

if __name__ == "__main__":
    main()
