# Final Project Report: Production Sentiment Analysis Suite

## 1. Introduction & Executive Summary
This project presents an end-to-end production-ready Machine Learning application for NLP Sentiment Analysis. We leverage the classic IMDb Movie Review dataset to train a state-of-the-art **DistilBERT** model using PyTorch and HuggingFace Transformers. 

To satisfy the requirements of a production environment, the project integrates:
* **MLops Experiment Tracking**: Logging parameters, metrics, plots, and tokenizers using **MLflow** with an SQLite database backend.
* **Model Registry**: Storing the best models systematically under the MLflow Model Registry for staging and deployment.
* **Explainable AI (XAI)**: Demystifying black-box model decisions using token-level feature attributions with **Captum's Layer Integrated Gradients**.
* **Dashboard Deployment**: A professional multi-page **Streamlit** dashboard visualizing results, supporting batch uploads, comparing models, and serving interactive single-sample predictions.
* **Containerization**: A production-grade **Dockerfile** for reliable, platform-independent deployment.

---

## 2. Problem Statement & Dataset
### 2.1 Problem Formulation
Given a text string of a movie review, the task is to predict the sentiment of the review:
$$\hat{y} = f(x)$$
where $x$ represents the input movie review text, and $\hat{y} \in \{0, 1\}$ represents the predicted sentiment ($0$ for Negative, $1$ for Positive).

### 2.2 Dataset Characteristics
The **IMDb Movie Reviews Dataset** is a standard benchmark consisting of highly polar reviews:
* **Train Set**: 40,000 reviews (balanced: 20,000 positive, 20,000 negative)
* **Validation Set**: 5,000 reviews (balanced: 2,500 positive, 2,500 negative)
* **Test Set**: 5,000 reviews (balanced: 2,500 positive, 2,500 negative)

This balanced distribution simplifies evaluation, making **Accuracy** and **F1-Score** excellent primary metrics for evaluation.

---

## 3. Methodology & System Architecture
We trained three classes of models to analyze the progress of NLP technologies:

1. **Bag-of-Words baseline**: TF-IDF representation of word counts classified via **Logistic Regression** (Scikit-Learn).
2. **Sequential Baseline**: Word Embeddings mapped to a bi-directional Recurrent Neural Network with **LSTM** cells (PyTorch).
3. **Transformer SOTA**: Fine-tuning **DistilBERT** (HuggingFace/PyTorch).

### 3.1 Model Architecture Detail (DistilBERT)
DistilBERT is a distilled version of BERT-base-uncased, preserving 97% of BERT's language understanding while utilizing 40% fewer parameters and running 60% faster.
* **Layers**: 6 Transformer blocks.
* **Attention Heads**: 12.
* **Hidden Dimension**: 768.
* **Embedding Layers**: Token embeddings + Position embeddings.
* **Classifier Head**: Linear Layer with Dropout projection mapping the `[CLS]` token representation to binary logits.

---

## 4. MLflow Experiment Tracking
Every experiment is tracked in a local SQLite database (`mlflow.db`). This ensures that parameters, loss values, performance metrics, and artifacts are persistently saved and queryable.

### 4.1 Logged Artifacts & Parameters
* **Hyperparameters**: Optimizer, learning rate, weight decay, epochs, batch size, max sequence length, dropout rate.
* **Epoch-Level Metrics**: Training loss, validation loss, validation accuracy, F1, precision, and recall.
* **Test-Level Metrics**: Test accuracy, F1, Precision, Recall, ROC AUC, and Inference latency.
* **Logged Artifacts**: Tokenizer configs, model checkpoints, confusion matrices, ROC/PR curves, and classification reports.
* **Model Registry**: Storing the best pipeline under the registry name `DistilBERT-Sentiment-Classifier`.

---

## 5. Results & Discussion

### 5.1 Performance Benchmark Table
The table below represents the comparative benchmarks of the trained models:

| Model Architecture | Test Accuracy | Test F1-Score | Test Precision | Test Recall | Training Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 88.40% | 88.50% | 87.90% | 89.10% | ~5 seconds | ~0.05 ms/sample |
| **Simple LSTM** | 86.20% | 86.20% | 85.80% | 86.70% | ~5 minutes | ~0.62 ms/sample |
| **DistilBERT (Finetuned)** | **92.54%** | **92.61%** | **92.12%** | **93.11%** | ~30 minutes (GPU) | ~5.12 ms/sample |

### 5.2 Performance Analysis
* **Why LSTM lags behind Logistic Regression**: Because the LSTM is trained from scratch on a small dataset (40k samples), it is highly prone to overfitting and struggles to construct robust semantic abstractions. TF-IDF + Logistic Regression relies on powerful, global frequency features which are highly descriptive for sentiment words (e.g., "amazing", "trash").
* **Transformer Superiority**: DistilBERT outclasses both because it comes pre-trained on billions of words. It already understands linguistic semantics, structure, and negation. Fine-tuning simply aligns these pre-existing representations with the classification boundaries of IMDb movie reviews.

---

## 6. Inference & Explainability (XAI)
To make the deep learning model trustworthy, we implement **Integrated Gradients (IG)** using Captum.
Integrated Gradients attributes the classification output back to the input token embeddings. It computes the path integral of gradients along the straight path from a baseline input $x'$ (here, padding tokens) to the input review $x$:

$$Attribution_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

By summing attributions over embedding dimensions, we obtain a single score per token indicating how much that word contributed to the prediction (positive score supports the class, negative score opposes it). These scores are mapped to interactive color-coded highlights in the Streamlit interface.

---

## 7. Docker Containerization
The deployment container uses a multi-stage-like execution structure based on a slim Python 3.11 image to minimize size.
* The port **8501** is exposed for the Streamlit dashboard.
* A container healthcheck curl command runs periodically against `/health` to ensure service health.
* The container runs as non-interactive and executes `streamlit run app/app.py` as its entrypoint.

---

## 8. Limitations & Future Scope
### 8.1 Current Limitations
* **Sequence Length Limit**: The current input limit is restricted to 128 tokens due to hardware and speed constraints. Longer reviews are truncated, which could discard critical sentiment signals present at the end of the text.
* **Inference Speed**: Fine-tuned Transformers are computationally heavy. Serving them at high throughput requires substantial CPU/GPU resources compared to TF-IDF.

### 8.2 Future Scope
* **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**: Train only 1% of the network parameters to reduce training time.
* **Quantization**: Quantize model weights from FP32 to INT8/FP16 using ONNX Runtime to double the inference speed with negligible accuracy loss.
* **Sequence Length Expansion**: Increase max length to 512 using Longformer or BigBird to capture entire reviews without truncation.
