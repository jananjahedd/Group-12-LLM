from __future__ import annotations

import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score

root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'distilbert.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logs_path, level=logging.INFO,
                    format=format_style)


class SarcasmDataset(Dataset):
    def __init__(self, tokens=None, labels=None) -> None:
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased'
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        tokens = self.tokens[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            tokens,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def create_subsets(
        self,
        val_size: float = 0.3,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None
    ) -> Tuple[SarcasmDataset, SarcasmDataset]:
        tokens = np.array(self.tokens)
        labels = np.array(self.labels)

        if train_indices is not None and val_indices is not None:
            tokens_t = tokens[train_indices]
            labels_t = labels[train_indices]
            tokens_v = tokens[val_indices]
            labels_v = labels[val_indices]
        else:
            tokens_t, tokens_v, labels_t, labels_v = train_test_split(
                tokens, labels, test_size=val_size, random_state=42
            )

        train_subset = SarcasmDataset(tokens_t, labels_t)
        val_subset = SarcasmDataset(tokens_v, labels_v)

        return train_subset, val_subset

    def reduce_dataset(self, data_size) -> SarcasmDataset:
        tokens = np.array(self.tokens)
        labels = np.array(self.labels)

        tokens, _, labels, _ = train_test_split(
            tokens, labels, train_size=data_size, random_state=42
        )

        data = SarcasmDataset(tokens, labels)

        return data


def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device("mps")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    accuracies = []
    logging.info("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                avg_loss = total_loss / total_samples
                accuracy = correct_predictions / total_samples

                tepoch.set_postfix(loss=avg_loss, accuracy=accuracy)

        avg_loss = total_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / len(train_loader.dataset)
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}"
            f"Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
        )

        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0

        logging.info("Starting validation...")

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", unit="batch") as vepoch:
                for batch in vepoch:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs.logits, dim=1)
                    correct_predictions += (preds == labels).sum().item()
                    total_samples += labels.size(0)

                    avg_val_loss = val_loss / total_samples
                    val_accuracy = correct_predictions / total_samples

                    vepoch.set_postfix(
                        val_loss=avg_val_loss,
                        val_accuracy=val_accuracy
                    )

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_predictions / len(val_loader.dataset)
        accuracies.append(val_accuracy)

        logging.info(
            f"Validation Loss: {avg_val_loss:.4f}"
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

    final_accuracy = sum(accuracies) / len(accuracies)
    logging.info(
        f"Training complete."
        f"Final average validation accuracy: {final_accuracy:.4f}"
    )

    return final_accuracy


def evaluate_model(model, test_dataset):
    device = torch.device("mps")
    model.to(device)

    model.eval()
    predictions = []
    true_labels = []
    all_logits = []
    all_probabilities = []

    test_loader = DataLoader(test_dataset, batch_size=32)

    logging.info("Starting evaluation...")

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, preds = torch.max(outputs.logits, dim=1)

                all_logits.extend(logits.cpu().numpy())

                probabilities = torch.nn.functional.softmax(
                    logits.clone().detach(), dim=1
                )[:, 1].cpu().numpy()
                all_probabilities.extend(probabilities.tolist())

                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

                test_accuracy = accuracy_score(true_labels, predictions)

                tepoch.set_postfix(batch_idx=batch_idx, accuracy=test_accuracy)

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary')
    roc_auc = roc_auc_score(true_labels, all_probabilities)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(true_labels, all_probabilities)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color='blue',
        label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    logging.info("Evaluation completed.")
    logging.info(
        f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
    )

    return accuracy, f1, roc_auc


def k_fold_cross_validation(
        model,
        data: SarcasmDataset,
        k_folds=8,
        num_epochs=1,
        learning_rate=3e-5
):
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_accuracies = []

    logging.info(f"Starting {k_folds}-fold cross-validation...")

    for fold, (train_indices, val_indices) in enumerate(kf.split(data, )):
        logging.info(f"Fold {fold + 1}/{k_folds}")

        training, validation = data.create_subsets(
            train_indices=train_indices,
            val_indices=val_indices
        )

        train_loader = DataLoader(training, batch_size=32, shuffle=True)
        val_loader = DataLoader(validation, batch_size=32)

        accuracy = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )

        fold_accuracies.append(accuracy)
        logging.info(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    avg_accuracy = np.mean(fold_accuracies)
    logging.info(
        f"Average Accuracy across {k_folds} folds: {avg_accuracy:.4f}"
    )

    return avg_accuracy


def objective(trial):
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
    r = trial.suggest_categorical("r", [8, 16, 32])
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.1, 0.2, 0.3])
    learning_rate = trial.suggest_categorical(
        "learning_rate",
        [1e-5, 2e-5, 3e-5]
    )

    logging.info(
        f"Trial {trial.number}: Testing with hyperparameters: "
        f"lora_alpha={lora_alpha}"
        f"r={r}, lora_dropout={lora_dropout}"
        f"learning_rate={learning_rate}"
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"]
    )

    logging.info("Creating subsets...")

    peft_data = train.reduce_dataset(data_size=0.1)
    training, validation = peft_data.create_subsets(val_size=0.3)

    train_loader = DataLoader(training, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation, batch_size=64)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2)

    logging.info("Getting the PEFT model...")

    lora_model = get_peft_model(model, lora_config)

    val_accuracy = train_model(
        lora_model,
        train_loader,
        val_loader,
        num_epochs=1,
        learning_rate=learning_rate
    )

    logging.info(
        f"Trial {trial.number}: Validation Accuracy = {val_accuracy:.4f}"
    )

    return val_accuracy


if __name__ == "__main__":
    train_df = pd.read_csv(
        'data/splits/train_data_distilbert.csv')
    test_df = pd.read_csv(
        'data/splits/test_data_distilbert.csv')

    train = SarcasmDataset(
        tokens=train_df['comment_tokenized'],
        labels=train_df['label']
    )
    test = SarcasmDataset(
        tokens=test_df['comment_tokenized'],
        labels=test_df['label']
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    print("Best hyperparameters:", study.best_params)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )

    model = get_peft_model(model, lora_config)

    train_accuracy = k_fold_cross_validation(model, train)

    accuracy, f1, roc_auc = evaluate_model(model, test)
