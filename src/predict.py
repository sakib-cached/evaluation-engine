import os
import time
import yaml
import argparse
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SentimentPredictor:
    def __init__(self, model_dir: str, device_str: str = "auto"):
        # Determine device
        if device_str == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
            
        logger.info(f"Loading Predictor on device: {self.device}")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")
            
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        
        # Identify the embeddings layer for Captum explainability
        if hasattr(self.model, "distilbert"):
            self.embeddings_layer = self.model.distilbert.embeddings
        elif hasattr(self.model, "bert"):
            self.embeddings_layer = self.model.bert.embeddings
        else:
            # Fallback to first child embeddings
            self.embeddings_layer = list(self.model.children())[0].embeddings
            
        # Define forward function for Captum
        def forward_func(input_ids, attention_mask=None):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            return logits
            
        self.lig = LayerIntegratedGradients(forward_func, self.embeddings_layer)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts the sentiment of a single review.
        """
        start_time = time.time()
        
        # Tokenize and prepare input tensors
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        inference_time = (time.time() - start_time) * 1000 # in milliseconds
        
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
        
        label_map = {0: "Negative", 1: "Positive"}
        
        return {
            "text": text,
            "label": label_map[predicted_class],
            "class_id": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "Negative": probs[0].item(),
                "Positive": probs[1].item()
            },
            "inference_time_ms": inference_time
        }

    def explain(self, text: str) -> List[Tuple[str, float]]:
        """
        Computes token-level attributions for the input text using Layer Integrated Gradients.
        Returns a list of (token, attribution_score) tuples.
        """
        # Tokenize input (without long padding for cleaner visualizer)
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Predict class index
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            
        # Create baseline input: [CLS] followed by [PAD]s followed by [SEP]
        ref_input_ids = torch.zeros_like(input_ids)
        ref_input_ids[0, 0] = self.tokenizer.cls_token_id
        ref_input_ids[0, -1] = self.tokenizer.sep_token_id
        
        # Run Captum Layer Integrated Gradients attribution
        # target=predicted_class identifies the logit index to attribute
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            additional_forward_args=(attention_mask,),
            target=predicted_class,
            return_convergence_delta=True
        )
        
        # Sum over the embedding dimensions
        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        
        # Normalize the attributions
        norm = torch.norm(attributions_sum)
        if norm > 0:
            attributions_sum = attributions_sum / norm
            
        # Convert IDs back to actual tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Zip tokens with attribution scores
        token_attributions = list(zip(tokens, attributions_sum.cpu().tolist()))
        return token_attributions

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a movie review")
    parser.add_argument("--text", type=str, required=True, help="Movie review text to predict")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    model_dir = config["training"]["save_dir"]
    device_str = config["training"].get("device", "auto")
    
    try:
        predictor = SentimentPredictor(model_dir, device_str)
        result = predictor.predict(args.text)
        print("\nPrediction Results:")
        print(f"Text: '{result['text']}'")
        print(f"Sentiment: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
        
        print("\nExplainability (Top Tokens Attribution for predicted class):")
        explanations = predictor.explain(args.text)
        # Sort by absolute attribution value
        sorted_exp = sorted(explanations, key=lambda x: abs(x[1]), reverse=True)
        for token, score in sorted_exp[:10]:
            print(f"Token: {token:12} | Attribution: {score:+.4f}")
            
    except Exception as e:
        logger.error(f"Error during prediction pipeline execution: {str(e)}")

if __name__ == "__main__":
    main()
