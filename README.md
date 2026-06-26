# 🎬 IMDb Sentiment Analysis Suite (End-to-End Production ML App)

An end-to-end, production-ready Natural Language Processing (NLP) application that fine-tunes a **DistilBERT** classification model on the IMDb Movie Review Dataset. The project incorporates model training, evaluation, MLOps experiment tracking (MLflow with SQLite), explainability (Captum Integrated Gradients), and a multi-page interactive Streamlit dashboard.

---

## 🚀 Key Features
1. **Finetuning Transformer Model**: Uses PyTorch and HuggingFace to train a DistilBERT classifier.
2. **Baselines Training**: Fast scripts to train TF-IDF + Logistic Regression and bidirectional LSTM architectures.
3. **MLflow Tracking**: Persistent SQLite database backend tracking and versioning every experiment (hyperparameters, metrics, plots, tokenizers, models).
4. **Model Registry**: Automatically registers the best-performing model checkpoints.
5. **Explainable AI (XAI)**: Captum-powered Layer Integrated Gradients visualizing word-level attribution maps in real-time.
6. **Multi-page Streamlit Dashboard**: Overview, Live Inference (with highlights), Diagnostic Performance, MLflow runs browser, Batch CSV predictions, and Model Comparisons.
7. **Docker Containerization**: Lightweight, platform-agnostic Dockerfile configuration for production deployments.

---

## 📂 Project Structure
```text
evaluation-engine/
├── README.md
├── final_report.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── configs/
│   └── config.yaml
├── src/
│   ├── data_download.py      # Verifies & downloads datasets
│   ├── train_baselines.py    # Trains Logistic Regression & LSTM
│   ├── train.py              # Finetunes DistilBERT & logs to MLflow
│   ├── evaluate.py           # Runs tests, outputs curves & logs to MLflow
│   └── predict.py            # Runs inference & Captum attributions
├── app/
│   └── app.py                # Multi-page Streamlit dashboard app
├── notebooks/
│   └── exploration.ipynb     # Jupyter notebook for exploration
├── models/                   # Local weights directory
├── data/                     # Location of Train/Valid/Test CSVs
└── artifacts/                # Local directory for generated plots
```

---

## ⚙️ Installation & Setup

### 1. Set Up Virtual Environment & Dependencies
Clone the repository, initialize your virtual environment, and install dependencies:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify and Load Dataset
Ensure the IMDb CSV files are verified and present in `data/`. If you are missing files, download them using:
```bash
python src/data_download.py
```

---

## 🏋️ Model Training & Operations

### 1. Train Baselines (Logistic Regression & LSTM)
Train baseline models and log their baseline results to the MLflow database:
```bash
python src/train_baselines.py --config configs/config.yaml
```

### 2. Finetune DistilBERT
Finetune the main DistilBERT Transformer. 
> [!TIP]
> Use `--quick-test` to execute training on a tiny subset of reviews to verify the pipeline ends successfully before running the full dataset.

* **Quick Validation Run**:
  ```bash
  python src/train.py --config configs/config.yaml --quick-test
  ```
* **Full Production Run**:
  ```bash
  python src/train.py --config configs/config.yaml
  ```

### 3. Run Model Evaluation
Compute comprehensive evaluation metrics (Accuracy, F1, ROC AUC, precision, recall) and plot diagnostic performance curves (saved locally to `artifacts/` and logged to MLflow):
```bash
python src/evaluate.py --config configs/config.yaml
# Run evaluation on quick test subsets if trained on quick test:
python src/evaluate.py --config configs/config.yaml --quick-test
```

### 4. CLI Inference & Explainability Test
You can run inference directly in the shell to test prediction and see word-level attributions:
```bash
python src/predict.py --text "This movie was absolutely spectacular, I loved it!"
```

---

## 📊 Dashboard & Monitoring

### 1. Start MLflow Tracking UI
To view experiment runs, compare logs, and browse model artifacts, launch the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
Open `http://localhost:5000` in your web browser.

### 2. Start the Streamlit Dashboard
Launch the multi-page client application dashboard locally:
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` to view your dashboard.

---

## 🐳 Docker Deployment

To build and launch the production containerized deployment:

```bash
# Build the Docker image
docker build -t sentiment-app:1.0 .

# Run the container exposing port 8501
docker run -p 8501:8501 sentiment-app:1.0
```
Access the application dashboard at `http://localhost:8501`.
