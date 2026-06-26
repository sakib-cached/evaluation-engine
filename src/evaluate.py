import os
import json
import yaml
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    roc_curve,
    precision_recall_curve
)

import mlflow
from tqdm import tqdm
from train import IMDbDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Evaluation Loop ---
def evaluate_model(model, data_loader, device) -> Tuple[List[int], List[float], List[int], float]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating on Test Set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            _, predicted = outputs.logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    inference_time = (time.time() - start_time) / len(data_loader.dataset)
    return all_preds, all_probs, all_labels, inference_time

# --- Plotting Utilities ---
def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path: str, auc_score: float):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, save_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="blue", lw=2, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(history_path: str, save_path: str):
    if not os.path.exists(history_path):
        logger.warning(f"History file not found at {history_path}. Skipping loss curves plot.")
        return
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Main Evaluation Pipeline ---
def run_evaluation(config: Dict[str, Any], quick_test: bool = False):
    # Setup MLflow SQLite tracking
    tracking_uri = config["mlflow"].get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config["mlflow"].get("experiment_name", "IMDb-Sentiment-Analysis")
    
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
    
    # Load Model and Tokenizer
    save_dir = config["training"]["save_dir"]
    if not os.path.exists(save_dir):
        logger.critical(f"Model path {save_dir} does not exist. Please train the model first.")
        return
        
    logger.info(f"Loading tokenizer and model from {save_dir}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(save_dir).to(device)
    
    # Load dataset
    test_path = config["data"]["test_path"]
    logger.info(f"Loading test set from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    # Subsetting for quick test or config
    test_size = 50 if quick_test else config["data"].get("val_sample_size")
    if test_size is not None:
        test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=42).reset_index(drop=True)
        logger.info(f"Subset Test size: {len(test_df)}")
        
    # Setup Dataset & DataLoader
    max_length = config["model"]["max_length"]
    test_dataset = IMDbDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length)
    
    batch_size = 4 if quick_test else config["training"]["batch_size"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Run evaluation
    preds, probs, labels, inf_time = evaluate_model(model, test_loader, device)
    
    # Calculate Metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs)
    clf_report = classification_report(labels, preds, target_names=["Negative", "Positive"])
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Average Inference Time per Sample: {inf_time*1000:.2f} ms")
    print("\nClassification Report:\n", clf_report)
    
    # Ensure artifacts and models folders exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Generate and save plots
    logger.info("Generating and saving performance plots...")
    plot_confusion_matrix(labels, preds, "artifacts/confusion_matrix.png")
    plot_roc_curve(labels, probs, "artifacts/roc_curve.png", auc)
    plot_precision_recall_curve(labels, probs, "artifacts/precision_recall_curve.png")
    
    history_file = os.path.join(save_dir, "history.json")
    plot_loss_curves(history_file, "artifacts/loss_curves.png")
    
    # Save Classification Report as text file
    with open("artifacts/classification_report.txt", "w") as f:
        f.write(clf_report)
        
    # Find the last active MLflow run for the experiment and log metrics/plots
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            if runs:
                run_id = runs[0].info.run_id
                logger.info(f"Found last run: {runs[0].info.run_name} (ID: {run_id}). Logging test metrics...")
                
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metric("test_accuracy", acc)
                    mlflow.log_metric("test_precision", precision)
                    mlflow.log_metric("test_recall", recall)
                    mlflow.log_metric("test_f1_score", f1)
                    mlflow.log_metric("test_roc_auc", auc)
                    mlflow.log_metric("inference_time_per_sample_sec", inf_time)
                    
                    # Log artifacts
                    mlflow.log_artifact("artifacts/confusion_matrix.png")
                    mlflow.log_artifact("artifacts/roc_curve.png")
                    mlflow.log_artifact("artifacts/precision_recall_curve.png")
                    if os.path.exists("artifacts/loss_curves.png"):
                        mlflow.log_artifact("artifacts/loss_curves.png")
                    mlflow.log_artifact("artifacts/classification_report.txt")
                    
                logger.info("Successfully logged test metrics and artifacts to MLflow.")
            else:
                logger.warning("No runs found in MLflow experiment. Metrics will not be logged.")
        else:
            logger.warning(f"Experiment {experiment_name} not found in MLflow. Metrics will not be logged.")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned DistilBERT on IMDb test set")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--quick-test", action="store_true", help="Run in quick test mode with tiny subsets")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    run_evaluation(cfg, quick_test=args.quick_test)
