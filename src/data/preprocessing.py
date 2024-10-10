"""
File: preprocessing.py
Authors: Andrei Medesan, Janan Jahed, and Alexandru Cernat
Description:
"""
import os
import kaggle
import zipfile

# Check if Kaggle credentials exist or if environment variables are used
if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    raise FileNotFoundError("Kaggle API key not found. Please follow the setup instructions in the README.")

# Download the dataset from Kaggle
dataset = 'danofer/sarcasm'
output_path = 'data/sarcasm.zip'

# Download dataset
kaggle.api.dataset_download_files(dataset, path='data/', unzip=False)

# Unzip the dataset
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall('data/')

print("Dataset downloaded and extracted to the 'data/' directory.")