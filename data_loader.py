import os
import shutil
import kagglehub

def load_datasets():
    """
    Automatically checks and downloads the necessary lightweight datasets from Kaggle
    to prepare for training/fine-tuning.
    """
    print("Checking and downloading datasets via KaggleHub...")
    
    # 1. Download Ball Detection Dataset (Tennis Analysis)
    ball_dataset_path = kagglehub.dataset_download("salmahazemm/tennis-analysis")
    print(f"Ball Detection dataset downloaded to: {ball_dataset_path}")
    
    # 2. Download Player Actions Dataset
    action_dataset_path = kagglehub.dataset_download("orvile/tennis-player-actions-dataset")
    print(f"Player Action dataset downloaded to: {action_dataset_path}")
    
    return ball_dataset_path, action_dataset_path

if __name__ == "__main__":
    load_datasets()