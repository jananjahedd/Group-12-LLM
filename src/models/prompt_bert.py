import logging
import torch
import ast
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from transformers import BertForSequenceClassification, BertTokenizer, PreTrainedTokenizer
from datasets import load_dataset


# Set up logging and paths
root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
plots_dir = root / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'bert_prompt2.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logs_path, level=logging.INFO, format=format_style)

# File paths for datasets
train_data_path = root / 'data' / 'splits' / 'train_data_bert.csv'
test_data_path = root / 'data' / 'splits' / 'test_data_bert.csv'

logging.info("Loading datasets...")
dataset = load_dataset('csv', data_files={'train': str(train_data_path), 'test': str(test_data_path)})

# Use 10% of the data
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(int(len(dataset['train']) * 0.1)))
small_test_dataset = dataset['test'].shuffle(seed=42).select(range(int(len(dataset['test']) * 0.1)))

logging.info("Dataset subset with 10% of the data created successfully.")

MultipleLists = Tuple[List[List[int]], List[List[int]], List[int]]


class SarcasmDataset(Dataset):
    def __init__(self, input_ids: List[List[int]],
                 attention_masks: List[List[int]],
                 labels: List[int]) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_data_with_prompt(data_df: pd.DataFrame,
                            tokenizer: PreTrainedTokenizer) -> MultipleLists:
    """Create input IDs, attention masks, and labels with prompts."""
    input_ids = []
    attention_masks = []
    labels = data_df['label'].tolist()

    # Define multiple prompts
    prompt_templates = [
        "Is this sentence sarcastic? [SEP] ",
        "Classify the following sentence as sarcastic or not: [SEP] ",
        "Is this sentence expressing sarcasm, why? [SEP] "
    ]

    for idx, row in data_df.iterrows():
        # randomly select a prompt for each example
        prompt_template = np.random.choice(prompt_templates)

        # pre-tokenize the prompt
        prompt_token_ids = tokenizer.encode(prompt_template, add_special_tokens=False)

        # extract the already tokenized comment data
        comment_token_ids = ast.literal_eval(row['comment_tokenized'])
        comment_attention_mask = [1] * len(comment_token_ids)

        # combine the prompt token IDs and comment token IDs
        full_input_ids = prompt_token_ids + comment_token_ids

        # adjust attention mask
        prompt_attention_mask = [1] * len(prompt_token_ids)
        full_attention_mask = prompt_attention_mask + comment_attention_mask

        # Append to the final input lists
        input_ids.append(full_input_ids)
        attention_masks.append(full_attention_mask)

    return input_ids, attention_masks, labels


def create_data_without_prompt(data_df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> MultipleLists:
    """Create input IDs, attention masks, and labels without prompts."""
    input_ids = []
    attention_masks = []
    labels = data_df['label'].tolist()

    for idx, row in data_df.iterrows():
        # extract the already tokenized comment data
        comment_token_ids = ast.literal_eval(row['comment_tokenized'])
        comment_attention_mask = [1] * len(comment_token_ids)

        # Append to the final input lists
        input_ids.append(comment_token_ids)
        attention_masks.append(comment_attention_mask)

    return input_ids, attention_masks, labels


def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 3, learning_rate: float = 2e-5,
                weight_decay: float = 0.01, plot_prefix: str = "bert") -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
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

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

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
        val_losses.append(avg_val_loss)
        val_accuracy = correct_predictions / len(val_loader.dataset)
        accuracies.append(val_accuracy)

        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # plot accuracy and loss
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plots_dir / f"{plot_prefix}_loss.png")

    plt.figure()
    plt.plot(epochs, accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plots_dir / f"{plot_prefix}_accuracy.png")

    return sum(accuracies) / len(accuracies)


def collate_fn(batch):
    """
    Custom collate function to pad the sequences dynamically.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input_ids and attention_mask to the maximum length in the batch
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)  # Stack labels

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': labels
    }



if __name__ == "__main__":
    small_train_df = pd.DataFrame(small_train_dataset)
    small_test_df = pd.DataFrame(small_test_dataset)

    # initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # create datasets with prompts (for prompt-engineered BERT)
    train_input_ids, train_attention_masks, train_labels = create_data_with_prompt(small_train_df, tokenizer)
    test_input_ids, test_attention_masks, test_labels = create_data_with_prompt(small_test_df, tokenizer)

    # create datasets without prompts (for original BERT)
    train_input_ids_no_prompt, train_attention_masks_no_prompt, train_labels_no_prompt = create_data_without_prompt(small_train_df, tokenizer)
    test_input_ids_no_prompt, test_attention_masks_no_prompt, test_labels_no_prompt = create_data_without_prompt(small_test_df, tokenizer)

    # create SarcasmDataset with and without prompts
    train_data_with_prompt = SarcasmDataset(train_input_ids, train_attention_masks, train_labels)
    test_data_with_prompt = SarcasmDataset(test_input_ids, test_attention_masks, test_labels)

    train_data_no_prompt = SarcasmDataset(train_input_ids_no_prompt, train_attention_masks_no_prompt, train_labels_no_prompt)
    test_data_no_prompt = SarcasmDataset(test_input_ids_no_prompt, test_attention_masks_no_prompt, test_labels_no_prompt)

    # DataLoader for training and validation
    train_loader_with_prompt = DataLoader(train_data_with_prompt, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader_with_prompt = DataLoader(test_data_with_prompt, batch_size=64, collate_fn=collate_fn)

    train_loader_no_prompt = DataLoader(train_data_no_prompt, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader_no_prompt = DataLoader(test_data_no_prompt, batch_size=64, collate_fn=collate_fn)

    # load the BERT model for sequence classification
    model_with_prompt = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model_original = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # train the model with prompt engineering
    train_model(model_with_prompt, train_loader_with_prompt, val_loader_with_prompt, num_epochs=3, learning_rate=2e-5, weight_decay=0.01, plot_prefix="bert_with_prompt")

    # train the original model without prompt engineering
    train_model(model_original, train_loader_no_prompt, val_loader_no_prompt, num_epochs=3, learning_rate=2e-5, weight_decay=0.01, plot_prefix="bert_original")
