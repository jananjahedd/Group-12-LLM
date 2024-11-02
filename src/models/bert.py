"""
Authors: Janan Jahed, Andrei Medesan, Alexandru Cenrat
File: bert.py
Description:The code contains the importing of the bert model along
with hyper parameetr tuning and fine tuning it to prepare for training
and testing for sarcasm detection
"""
import os
import logging
import torch
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import optuna
import numpy as np

root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'bert.log'

logging.basicConfig(
    filename=str(logs_path),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

model_id = "bert-large-uncased"
logging.info("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(model_id)
logging.info("BERT model loaded successfully.")

tokenizer = BertTokenizer.from_pretrained(model_id)
logging.info("BERT tokenizer loaded successfully.")

train_data_path = root / 'data' / 'splits' / 'train_data_bert.csv'
test_data_path = root / 'data' / 'splits' / 'test_data_bert.csv'

logging.info("Loading datasets...")
dataset = load_dataset('csv', data_files={'train': str(train_data_path),
                                          'test': str(test_data_path)})


def prepare_features(example):
    """
    Prepares the features from a dataset example for input to the BERT model

    args:
        e.g: dict: a dictionary containing the 'comment' and 'label'

    returns:
        dict: a dictionary containing input_ids, attention_mask, and labels
    """
    encoding = tokenizer(
        example['comment'],
        padding='max_length',
        truncation=True,
        max_length=430
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': example['label']
    }


dataset = dataset.map(prepare_features)

logging.info("Dataset loaded and prepared successfully.")


def fast_evaluation(trainer, dataset):
    """
    Evaluates the model's performance on the provided dataset

    args:
        trainer: the trainer instance for the model
        dataset: the dataset to evaluate on

    returns:
        tuple: accuracy, F1 score, and error rate
    """
    predictions_output = trainer.predict(dataset)
    predictions = predictions_output.predictions
    true_labels = predictions_output.label_ids
    preds = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")
    error_rate = 1 - accuracy

    return accuracy, f1, error_rate


def objective(trial):
    """
    Defines the objective function for hyperparameter tuning

    args:
        trial (optuna.Trial): the Optuna trial object to sample hyperparameters

    returns:
        float: the evaluation loss for the current set of hyperparameters
    """

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 3e-5, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
    per_device_train_batch_size = trial.suggest_categorical(
        'per_device_train_batch_size', [8, 16])

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=weight_decay,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    logging.info("Creating Trainer instance for hyperparameter tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_loss']


logging.info("Starting hyperparameter tuning with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)

best_trial = study.best_trial
logging.info(f"Best trial: {best_trial.params}")


best_training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_trial.params['learning_rate'],
    per_device_train_batch_size=best_trial.params[
        'per_device_train_batch_size'],
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=best_trial.params['weight_decay'],
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

logging.info("Evaluating the pre-trained BERT model" +
             "with best hyperparameters...")
trainer = Trainer(
    model=model,
    args=best_training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

logging.info("Evaluating the pre-trained BERT model...")
original_results = trainer.evaluate()
logging.info("Original BERT Results:\n%s", original_results)

accuracy, f1, error_rate = fast_evaluation(trainer, dataset['test'])
logging.info(f"Original BERT Accuracy: {accuracy:.4f}" +
             f"F1 Score: {f1:.4f}, Error Rate: {error_rate:.4f}")

logging.info("Configuring LoRA for fine-tuning...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.2,
    task_type="SEQ_CLS",
)
logging.info("LoRA configuration complete.")

logging.info("Getting LoRA model...")
lora_model = get_peft_model(model, lora_config)
logging.info("LoRA model obtained successfully.")

logging.info("Creating Trainer instance for LoRA fine-tuning...")
lora_trainer = Trainer(
    model=lora_model,
    args=best_training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

logging.info("Fine-tuning the LoRA model...")
lora_trainer.train()
logging.info("Fine-tuning complete.")

logging.info("Evaluating the LoRA fine-tuned model...")
lora_results = lora_trainer.evaluate()
logging.info("LoRA Fine-Tuned BERT Results:\n%s", lora_results)

accuracy, f1, error_rate = fast_evaluation(lora_trainer, dataset['test'])
logging.info(f"LoRA Fine-Tuned BERT Accuracy: {accuracy:.4f}" +
             f"F1 Score: {f1:.4f}, Error Rate: {error_rate:.4f}")

logging.info("Creating performance comparison DataFrame...")
performance_comparison = pd.DataFrame({
    "Metric": original_results.keys(),
    "Original BERT": original_results.values(),
    "LoRA Fine-Tuned BERT": lora_results.values()
})

performance_comparison["Accuracy"] = [accuracy] * len(performance_comparison)
performance_comparison["F1 Score"] = [f1] * len(performance_comparison)
performance_comparison["Error Rate"] = (
    [error_rate] * len(performance_comparison))

logging.info("Saving performance comparison to CSV...")
performance_save_path = root / 'results' / 'performance_comparison_updated.csv'
performance_comparison.to_csv(performance_save_path, index=False)
logging.info("Performance comparison saved successfully.")

metrics = list(original_results.keys()) + ["Accuracy",
                                           "F1 Score", "Error Rate"]
original_values = list(original_results.values()) + [
    accuracy, f1, error_rate]
lora_values = list(lora_results.values()) + [accuracy, f1, error_rate]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(metrics))

plt.bar(index, original_values, bar_width, label='Original BERT')
plt.bar([i + bar_width for i in index], lora_values, bar_width,
        label='LoRA Fine-Tuned BERT')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Comparison: Original BERT vs. LoRA Fine-Tuned BERT')
plt.xticks([i + bar_width/2 for i in index], metrics, rotation=45)
plt.legend()

plot_save_path = root / 'results' / 'performance_comparison.png'
plt.tight_layout()
plt.savefig(plot_save_path)
logging.info(f"Performance comparison plot saved at {plot_save_path}")
plt.show()


def get_predictions_and_labels(trainer, dataset):
    predictions_output = trainer.predict(dataset)
    predictions = predictions_output.predictions
    true_labels = predictions_output.label_ids

    probs = torch.softmax(torch.tensor(predictions), dim=-1).numpy()
    positive_class_probs = probs[:, 1]

    return positive_class_probs, true_labels


logging.info("Getting predictions and labels for ROC/AUC...")
original_probs, original_labels = get_predictions_and_labels(trainer,
                                                             dataset['test'])
lora_probs, lora_labels = get_predictions_and_labels(lora_trainer,
                                                     dataset['test'])

fpr_original, tpr_original, _ = roc_curve(original_labels, original_probs)
roc_auc_original = auc(fpr_original, tpr_original)

fpr_lora, tpr_lora, _ = roc_curve(lora_labels, lora_probs)
roc_auc_lora = auc(fpr_lora, tpr_lora)

plt.figure()
plt.plot(fpr_original, tpr_original, color='blue',
         lw=2, label='Original BERT ROC (AUC = %0.2f)' % roc_auc_original)
plt.plot(fpr_lora, tpr_lora, color='orange', lw=2,
         label='LoRA Fine-Tuned BERT ROC (AUC = %0.2f)' % roc_auc_lora)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Original BERT vs. LoRA Fine-Tuned BERT')
plt.legend(loc="lower right")

roc_plot_save_path = root / 'results' / 'roc_comparison.png'
plt.tight_layout()
plt.savefig(roc_plot_save_path)
logging.info(f"ROC comparison plot saved at {roc_plot_save_path}")
plt.show()

logging.info(f"Original BERT AUC: {roc_auc_original:.4f}")
logging.info(f"LoRA Fine-Tuned BERT AUC: {roc_auc_lora:.4f}")
