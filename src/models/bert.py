import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import BertForSequenceClassification, BertTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split

# Set up logging and paths
root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'bert.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logs_path, level=logging.INFO, format=format_style)

# File paths (same as previous code)
train_data_path = root / 'data' / 'processed' / 'train_data.csv'
test_data_path = root / 'data' / 'processed' / 'test_data.csv'

class SarcasmDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        label = self.labels[idx]

        inputs = tokenizer(
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

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
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
                logging.info(f"Batch Loss: {loss.item():.4f}, Running Accuracy: {accuracy:.4f}")

        avg_loss = total_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / len(train_loader.dataset)
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0 

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_predictions / len(val_loader.dataset)
        accuracies.append(val_accuracy)

        logging.info(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    final_accuracy = sum(accuracies) / len(accuracies)
    logging.info(f"Training complete. Final average validation accuracy: {final_accuracy:.4f}")

    return final_accuracy

def create_subsets(data: SarcasmDataset, val_size):
    tokens = np.array(data.tokens)
    labels = np.array(data.labels)

    tokens_train, tokens_val, labels_train, labels_val = train_test_split(
        tokens, labels, test_size=val_size, random_state=42
    )

    train_subset = SarcasmDataset(tokens_train, labels_train)
    val_subset = SarcasmDataset(tokens_val, labels_val)

    return train_subset, val_subset

def reduce_dataset(data: SarcasmDataset, data_size):
    tokens = np.array(data.tokens)
    labels = np.array(data.labels)

    tokens, _, labels, _ = train_test_split(
        tokens, labels, train_size=data_size, random_state=42
    )

    data = SarcasmDataset(tokens, labels)

    if hasattr(data, 'reset_index'):
        data.reset_index(drop=True, inplace=True)

    return data

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
        f"lora_alpha={lora_alpha}, r={r}, lora_dropout={lora_dropout}, learning_rate={learning_rate}"
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"]
    )

    logging.info("Creating subsets...")

    training, validation = create_subsets(peft_data, val_size=0.3)

    train_loader = DataLoader(training, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation, batch_size=64)

    logging.info("Getting the PEFT model...")

    lora_model = get_peft_model(model, lora_config)

    val_accuracy = train_model(
        lora_model,
        train_loader,
        val_loader,
        num_epochs=1,
        learning_rate=learning_rate
    )

    logging.info(f"Trial {trial.number}: Validation Accuracy = {val_accuracy:.4f}")

    return val_accuracy

if __name__ == "__main__":
    # Loading the datasets from the initialized paths
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_df['features'] = train_df.drop(columns=['label']).astype(str).agg(' '.join, axis=1)
    test_df['features'] = test_df.drop(columns=['label']).astype(str).agg(' '.join, axis=1)

    train = SarcasmDataset(
        tokens=train_df['features'],
        labels=train_df['label']
    )
    test = SarcasmDataset(
        tokens=test_df['features'],
        labels=test_df['label']
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)

    peft_data = reduce_dataset(train, data_size=0.1)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    print("Best hyperparameters:", study.best_params)
