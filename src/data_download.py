import os
import logging
import pandas as pd
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def verify_dataset(data_dir: str = "data") -> Dict[str, bool]:
    """
    Verifies that the train, validation, and test datasets exist in the data folder.
    Returns a dictionary indicating the status of each file.
    """
    required_files = {
        "train": os.path.join(data_dir, "Train.csv"),
        "valid": os.path.join(data_dir, "Valid.csv"),
        "test": os.path.join(data_dir, "Test.csv")
    }
    
    status = {}
    for name, path in required_files.items():
        if os.path.exists(path):
            try:
                # Load first few lines to verify schema
                df = pd.read_csv(path, nrows=5)
                if "text" in df.columns and "label" in df.columns:
                    status[name] = True
                    logger.info(f"Verified: {path} exists and has valid schema (columns: {list(df.columns)}).")
                else:
                    status[name] = False
                    logger.warning(f"Invalid schema: {path} is missing 'text' or 'label' columns.")
            except Exception as e:
                status[name] = False
                logger.error(f"Error reading {path}: {str(e)}")
        else:
            status[name] = False
            logger.warning(f"Missing: {path} does not exist.")
            
    return status

def download_dataset(data_dir: str = "data"):
    """
    Fallback method to download dataset if files are missing.
    Downloads the IMDb dataset from a reliable public source.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for raw IMDB CSV dataset on GitHub
    imdb_url = "https://raw.githubusercontent.com/AnirudhDhanda/IMDB-Movie-Reviews-Dataset/master/IMDB%20Dataset.csv"
    
    logger.info("Attempting to download IMDb dataset from fallback URL...")
    try:
        df = pd.read_csv(imdb_url)
        logger.info(f"Successfully downloaded dataset with shape: {df.shape}")
        
        # Standardize columns and labels if needed
        # Expected: text (str) and label (0 for negative, 1 for positive)
        if "review" in df.columns:
            df = df.rename(columns={"review": "text"})
        if "sentiment" in df.columns:
            # Convert positive/negative text to 1/0
            df["label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
            df = df.drop(columns=["sentiment"])
            
        # Shuffle and split: 40k train, 5k valid, 5k test
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df.iloc[:40000]
        valid_df = df.iloc[40000:45000]
        test_df = df.iloc[45000:50000]
        
        train_df.to_csv(os.path.join(data_dir, "Train.csv"), index=False)
        valid_df.to_csv(os.path.join(data_dir, "Valid.csv"), index=False)
        test_df.to_csv(os.path.join(data_dir, "Test.csv"), index=False)
        
        logger.info("Successfully split and saved dataset to Train.csv, Valid.csv, and Test.csv.")
    except Exception as e:
        logger.critical(f"Failed to download IMDb dataset: {str(e)}")
        raise e

def main():
    data_dir = "data"
    status = verify_dataset(data_dir)
    
    if not all(status.values()):
        logger.warning("One or more dataset files are missing or invalid. Starting download...")
        download_dataset(data_dir)
        verify_dataset(data_dir)
    else:
        logger.info("All dataset files verified successfully! Ready for training.")

if __name__ == "__main__":
    main()
