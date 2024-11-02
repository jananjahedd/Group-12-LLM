"""
Authors: Janan Jahed, Andrei Medesan, Alexandru Cenrat
File: prompt_distilbert.py
Description: Training baseline and prompt engineered
DistilBERT models on the sarcasm dataset.
Sources for the code implementation:
https://huggingface.co/docs/transformers/en/model_doc/distilbert
https://tqdm.github.io
https://pypi.org/project/torch/
"""
import logging
import torch
import ast
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from datasets import load_dataset


# set up logging and paths
root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
plots_dir = root / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'distilbert_prompt.log'
logging.basicConfig(filename=logs_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# paths to datasets
train_data_path = root / 'data' / 'splits' / 'train_data_distilbert.csv'
test_data_path = root / 'data' / 'splits' / 'test_data_distilbert.csv'

# load Hugging Face datasets
logging.info("Loading datasets...")
dataset = load_dataset('csv', data_files={'train': str(train_data_path),
                                          'test': str(test_data_path)})

# use 10% of the data for training and testing
small_train_dataset = dataset['train'].shuffle(seed=42).select(
    range(int(len(dataset['train']) * 0.1)))
small_test_dataset = dataset['test'].shuffle(seed=42).select(
    range(int(len(dataset['test']) * 0.1)))

logging.info("Dataset subset with 10% of the data created successfully.")

# drop unnecessary columns
columns_to_drop = ['score', 'ups', 'downs', 'comment_length',
                   'parent_comment_length']
small_train_dataset = small_train_dataset.drop(columns=columns_to_drop)
small_test_dataset = small_test_dataset.drop(columns=columns_to_drop)

logging.info("Dataset subset with 10% of the data created successfully.")

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SarcasmDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]],
                 tokenizer: DistilBertTokenizer,
                 use_prompts: bool = True) -> None:
        self.input_ids = []
        self.attention_masks = []
        self.labels = [example['label'] for example in data_list]

        prompts = [
            "Is this sentence sarcastic or not? [MASK] [SEP] ",
            "Label this comment as sarcastic or not: [MASK] [SEP] ",
            "Determine sarcasm: [MASK] [SEP] "
        ] if use_prompts else [""]

        for example in data_list:
            prompt = np.random.choice(prompts)
            prompt_token_ids = tokenizer.encode(prompt,
                                                add_special_tokens=False)
            comment_token_ids = ast.literal_eval(example['comment_tokenized'])
            self.input_ids.append(prompt_token_ids + comment_token_ids)
            self.attention_masks.append([1] * len(self.input_ids[-1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx],
                                           dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def collate_fn(
        batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True,
                                    padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True,
                                          padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'labels': labels
    }


# Training function with cross-validation
def train_model_with_cv(dataset: Dataset, model: nn.Module,
                        tokenizer: DistilBertTokenizer, num_epochs: int = 2,
                        learning_rate: float = 2e-5,
                        weight_decay: float = 0.01, k_folds: int = 3,
                        prefix: str = "") -> None:
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logging.info(f"Fold {fold+1}/{k_folds}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=32,
                                  sampler=train_subsampler,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(dataset, batch_size=64,
                                sampler=val_subsampler,
                                collate_fn=collate_fn)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses, val_accuracies, val_roc_aucs = [], [], [], []

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct_preds, total_preds = 0, 0
            script = f"Training Fold {fold+1} Epoch {epoch+1}/{num_epochs}"

            for batch in tqdm(train_loader, desc=script):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss, val_preds, val_labels = 0, [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    val_loss += criterion(outputs.logits, labels).item()
                    val_preds.extend(torch.argmax(outputs.logits,
                                                  dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)

            accuracy = accuracy_score(val_labels, val_preds)
            roc_auc = roc_auc_score(label_binarize(val_labels, classes=[0, 1]),
                                    label_binarize(val_preds, classes=[0, 1]))
            val_accuracies.append(accuracy)
            val_roc_aucs.append(roc_auc)

            logging.info(f"{prefix} Fold {fold+1} - Epoch {epoch+1} - " +
                         f"Train Loss: {avg_train_loss:.4f} - Val Loss: " +
                         f"{avg_val_loss:.4f} - Val Accuracy: {accuracy:.4f}" +
                         f" - Val ROC AUC: {roc_auc:.4f}")

        # plot metrics
        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.legend()
        plt.title(f"{prefix} Training and Validation Loss (Fold {fold+1})")
        plt.savefig(plots_dir / f"{prefix}_fold_{fold+1}_loss.png")

        plt.figure()
        plt.plot(range(1, num_epochs + 1), val_accuracies,
                 label='Validation Accuracy')
        plt.plot(range(1, num_epochs + 1), val_roc_aucs,
                 label='Validation ROC AUC')
        plt.legend()
        plt.title(f"{prefix} Validation Metrics (Fold {fold+1})")
        plt.savefig(plots_dir / f"{prefix}_fold_{fold+1}_metrics.png")


if __name__ == "__main__":
    # initialize tokenized data for DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset_with_prompt = SarcasmDataset(small_train_dataset, tokenizer,
                                         use_prompts=True)
    dataset_without_prompt = SarcasmDataset(small_train_dataset, tokenizer,
                                            use_prompts=False)

    # initialize the models
    model_with_prompt = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2).to(device)
    model_baseline = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2).to(device)

    # train models on the data
    train_model_with_cv(dataset_with_prompt, model_with_prompt, tokenizer,
                        prefix="prompt_distilbert")
    train_model_with_cv(dataset_without_prompt, model_baseline, tokenizer,
                        prefix="distilbert")
