import os
import time
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import mlflow
import mlflow.pytorch
import mlflow.transformers
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PyTorch Dataset class ---
class IMDbDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# --- Early Stopping Helper ---
class EarlyStopping:
    def __init__(self, patience: int = 2, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

# --- Validation function ---
def evaluate_epoch(model, data_loader, device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item() * input_ids.size(0)
            
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            _, predicted = outputs.logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader.dataset)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }
    
    return avg_loss, metrics

# --- Main Training Pipeline ---
def run_training(config: Dict[str, Any], quick_test: bool = False):
    # Set tracking URI and experiment
    mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(config["mlflow"].get("experiment_name", "IMDb-Sentiment-Analysis"))
    
    # Determine device
    device_config = config["training"].get("device", "auto")
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)
    logger.info(f"Using device: {device}")
    
    # Load dataset paths
    train_path = config["data"]["train_path"]
    valid_path = config["data"]["valid_path"]
    
    logger.info("Loading datasets...")
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    
    # Subsetting for quick test or if sample size is configured
    train_size = 200 if quick_test else config["data"].get("train_sample_size")
    val_size = 50 if quick_test else config["data"].get("val_sample_size")
    
    if train_size is not None:
        train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=42).reset_index(drop=True)
        logger.info(f"Subset Train size: {len(train_df)}")
    if val_size is not None:
        valid_df = valid_df.sample(n=min(val_size, len(valid_df)), random_state=42).reset_index(drop=True)
        logger.info(f"Subset Valid size: {len(valid_df)}")
        
    # Setup Tokenizer
    model_name = config["model"]["model_name"]
    max_length = config["model"]["max_length"]
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    # Setup Datasets & DataLoaders
    train_dataset = IMDbDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    valid_dataset = IMDbDataset(valid_df["text"].tolist(), valid_df["label"].tolist(), tokenizer, max_length)
    
    batch_size = 4 if quick_test else config["training"]["batch_size"]
    epochs = 2 if quick_test else config["training"]["epochs"]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Load Model
    logger.info(f"Loading model: {model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=config["model"]["num_classes"],
        seq_classif_dropout=config["model"].get("dropout", 0.2)
    ).to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config["training"]["learning_rate"]), 
        eps=float(config["training"]["adam_epsilon"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["training"].get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config["training"].get("early_stopping_patience", 2))
    
    run_name = "DistilBERT_Finetune_Quick" if quick_test else "DistilBERT_Finetune"
    
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run.info.run_name} (ID: {run.info.run_id})")
        
        # Log Hyperparameters
        mlflow.log_params({
            "model_name": model_name,
            "max_sequence_length": max_length,
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": config["training"]["weight_decay"],
            "dropout": config["model"]["dropout"],
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "train_samples": len(train_dataset),
            "val_samples": len(valid_dataset)
        })
        
        best_val_loss = float("inf")
        save_dir = config["training"]["save_dir"]
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
        start_time = time.time()
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item() * input_ids.size(0)
                progress_bar.set_postfix({"loss": loss.item()})
                
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            
            # Validate
            avg_val_loss, val_metrics = evaluate_epoch(model, valid_loader, device)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Append history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1_score"])
            
            # Log epoch metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
            mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
            mlflow.log_metric("val_f1_score", val_metrics["f1_score"], step=epoch)
            mlflow.log_metric("val_roc_auc", val_metrics["roc_auc"], step=epoch)
            
            # Checkpoint model if best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"Saving best model checkpoint to local directory: {save_dir}")
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                
            # Early Stopping
            if early_stopping(avg_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_sec", training_time)
        logger.info(f"Training completed in {training_time:.2f} seconds.")
        
        # Save history.json
        import json
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

        
        # Log best checkpoint to MLflow Registry and Artifacts
        logger.info("Logging best model to MLflow registry...")
        
        # Load best model for logging
        best_model = DistilBertForSequenceClassification.from_pretrained(save_dir).to(device)
        
        # Log model & tokenizer bundle as HuggingFace pipeline in MLflow
        try:
            import transformers
            sentiment_pipeline = transformers.pipeline(
                "text-classification", 
                model=best_model, 
                tokenizer=tokenizer,
                device=0 if device.type == "cuda" else -1
            )
            
            # Use transformers flavor for logging
            mlflow.transformers.log_model(
                transformers_model=sentiment_pipeline,
                artifact_path="model",
                registered_model_name=config["mlflow"]["registry_model_name"]
            )
            logger.info("Successfully registered model using MLflow Transformers flavor.")
        except Exception as e:
            logger.error(f"Failed to log with MLflow Transformers flavor: {e}. Falling back to standard PyTorch logging...")
            mlflow.pytorch.log_model(
                best_model, 
                artifact_path="model",
                registered_model_name=config["mlflow"]["registry_model_name"],
                serialization_format="pickle"
            )
            
        # Log tokenizer files as separate artifacts
        mlflow.log_artifacts(save_dir, artifact_path="tokenizer_files")
        
        logger.info(f"Training and registration complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune DistilBERT on IMDb Sentiment dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--quick-test", action="store_true", help="Run in quick test mode with tiny subsets")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    run_training(cfg, quick_test=args.quick_test)
