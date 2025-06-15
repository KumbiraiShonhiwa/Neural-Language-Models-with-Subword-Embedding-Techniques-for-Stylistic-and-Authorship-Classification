# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
import json
import wandb
from datetime import datetime

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    # Paths
    data_dir = "/content/drive/MyDrive/preprocessed_output"
    output_dir = "/content/drive/MyDrive/authorship_models"

    # Model selection (choose one: 'bpe', 'wordpiece', 'sentencepiece')
    tokenization_method = 'sentencepiece'  # Change this to switch between methods

    # Model configurations
    model_configs = {
        'bpe': {
            'model_name': 'xlm-roberta-base',
            'train_data': 'bpe_train_dataset.pt',
            'val_data': 'bpe_val_dataset.pt',
            'test_data': 'bpe_test_dataset.pt'
        },
        'wordpiece': {
            'model_name': 'bert-base-multilingual-cased',
            'train_data': 'wordpiece_train_dataset.pt',
            'val_data': 'wordpiece_val_dataset.pt',
            'test_data': 'wordpiece_test_dataset.pt'
        },
        'sentencepiece': {
            'model_name': 'xlm-roberta-large',
            'train_data': 'sentencepiece_train_dataset.pt',
            'val_data': 'sentencepiece_val_dataset.pt',
            'test_data': 'sentencepiece_test_dataset.pt'
        }
    }

    # Training hyperparameters
    learning_rate = 5e-5
    batch_size = 8
    num_epochs = 20
    warmup_steps = 500
    weight_decay = 0.01
    max_length = 128

    # Training settings
    gradient_accumulation_steps = 4
    eval_steps = 100
    save_steps = 200
    logging_steps = 50

    # Early stopping
    early_stopping_patience = 3

    # Experiment tracking
    use_wandb = True
    project_name = "twitter-authorship-attribution"

config = Config()

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

# Initialize wandb if enabled
if config.use_wandb:
    wandb.init(
        project=config.project_name,
        name=f"{config.tokenization_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "tokenization_method": config.tokenization_method,
            "model_name": config.model_configs[config.tokenization_method]['model_name'],
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs
        }
    )

# Custom Dataset Class
class AuthorshipDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Load datasets
print(f"Loading {config.tokenization_method} datasets...")
model_config = config.model_configs[config.tokenization_method]

train_data = torch.load(os.path.join(config.data_dir, model_config['train_data']))
val_data = torch.load(os.path.join(config.data_dir, model_config['val_data']))
test_data = torch.load(os.path.join(config.data_dir, model_config['test_data']))

# Get number of authors (classes)
num_authors = len(set(train_data['labels'].tolist()))
print(f"Number of authors: {num_authors}")

# Create datasets
train_dataset = AuthorshipDataset(
    train_data['input_ids'],
    train_data['attention_mask'],
    train_data['labels']
)

val_dataset = AuthorshipDataset(
    val_data['input_ids'],
    val_data['attention_mask'],
    val_data['labels']
)

test_dataset = AuthorshipDataset(
    test_data['input_ids'],
    test_data['attention_mask'],
    test_data['labels']
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Load model and tokenizer
print(f"Loading model: {model_config['model_name']}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_config['model_name'],
    num_labels=num_authors,
    ignore_mismatched_sizes=True
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])

# Define metrics
def compute_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Get predicted classes
    preds = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )

    # Top-5 accuracy
    top_5_preds = np.argsort(predictions, axis=1)[:, -5:]
    top_5_acc = np.mean([label in top_5_preds[i] for i, label in enumerate(labels)])

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'top_5_accuracy': top_5_acc
    }

    return metrics

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(config.output_dir, f"{config.tokenization_method}_model"),
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=config.eval_steps,
    save_strategy="no",  # Don't save during training
    # save_steps=config.save_steps,  
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=config.num_epochs,
    weight_decay=config.weight_decay,
    warmup_steps=config.warmup_steps,
    logging_dir=os.path.join(config.output_dir, "logs"),
    logging_steps=config.logging_steps,
    load_best_model_at_end=False,  # Changed to False since we're not saving checkpoints
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    fp16=True,  # Enable mixed precision training
    report_to="wandb" if config.use_wandb else "none",
    run_name=f"{config.tokenization_method}_authorship",
    # save_total_limit=3,  # Comment this out
    push_to_hub=False,
    do_train=True,  # Added
    do_eval=True,   # Added
)
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
)

# Train the model
print("\nStarting training...")
train_result = trainer.train()

# Save the final model
trainer.save_model()
trainer.save_state()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)

# Save results
results = {
    'tokenization_method': config.tokenization_method,
    'model_name': model_config['model_name'],
    'train_results': train_result.metrics,
    'test_results': test_results,
    'num_authors': num_authors,
    'training_samples': len(train_dataset),
    'validation_samples': len(val_dataset),
    'test_samples': len(test_dataset)
}

with open(os.path.join(config.output_dir, f"{config.tokenization_method}_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# Generate detailed predictions for error analysis
print("\nGenerating detailed predictions...")
test_predictions = trainer.predict(test_dataset)
predictions = np.argmax(test_predictions.predictions, axis=1)
true_labels = test_predictions.label_ids

# Calculate per-author metrics
from collections import defaultdict
author_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})

for pred, true in zip(predictions, true_labels):
    author_metrics[true]['total'] += 1
    if pred == true:
        author_metrics[true]['correct'] += 1

# Calculate per-author accuracy
author_accuracies = {
    author: metrics['correct'] / metrics['total']
    for author, metrics in author_metrics.items()
}

# Plot confusion matrix (for a subset of authors if too many)
num_authors_to_plot = min(20, num_authors)
if num_authors > num_authors_to_plot:
    # Select top authors with most errors
    author_errors = [(author, 1 - acc) for author, acc in author_accuracies.items()]
    author_errors.sort(key=lambda x: x[1], reverse=True)
    authors_to_plot = [x[0] for x in author_errors[:num_authors_to_plot]]

    # Filter predictions and labels for selected authors
    mask = np.isin(true_labels, authors_to_plot)
    filtered_predictions = predictions[mask]
    filtered_labels = true_labels[mask]

    # Create mapping for plotting
    author_mapping = {author: i for i, author in enumerate(authors_to_plot)}
    filtered_predictions_mapped = [author_mapping.get(p, -1) for p in filtered_predictions]
    filtered_labels_mapped = [author_mapping.get(l, -1) for l in filtered_labels]
else:
    filtered_predictions_mapped = predictions
    filtered_labels_mapped = true_labels
    authors_to_plot = list(range(num_authors))

# Create confusion matrix
cm = confusion_matrix(filtered_labels_mapped, filtered_predictions_mapped)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Confusion Matrix - {config.tokenization_method.upper()} Model')
plt.xlabel('Predicted Author')
plt.ylabel('True Author')
plt.tight_layout()
plt.savefig(os.path.join(config.output_dir, f'{config.tokenization_method}_confusion_matrix.png'))
plt.close()

# Plot per-author accuracy
plt.figure(figsize=(15, 8))
sorted_authors = sorted(author_accuracies.items(), key=lambda x: x[1])
authors, accuracies = zip(*sorted_authors[:30])  # Show worst 30 authors
plt.bar(range(len(authors)), accuracies)
plt.xlabel('Author ID')
plt.ylabel('Accuracy')
plt.title(f'Per-Author Accuracy (Worst 30) - {config.tokenization_method.upper()}')
plt.xticks(range(len(authors)), authors, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(config.output_dir, f'{config.tokenization_method}_per_author_accuracy.png'))
plt.close()

# Advanced analysis functions
def analyze_errors(predictions, true_labels, test_data, author_map):
    """Analyze prediction errors in detail"""
    errors = []

    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != true:
            errors.append({
                'index': i,
                'true_author': true,
                'predicted_author': pred,
                'text_length': test_data['input_ids'][i].ne(0).sum().item()
            })

    return errors

def get_most_confused_pairs(cm, n=10):
    """Find the most confused author pairs"""
    confused_pairs = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    return confused_pairs[:n]

# Perform error analysis
errors = analyze_errors(predictions, true_labels, test_data, train_data['author_map'])
error_rate = len(errors) / len(true_labels)
print(f"\nError rate: {error_rate:.2%}")

# Find most confused author pairs
confused_pairs = get_most_confused_pairs(cm)
print("\nMost confused author pairs:")
for true_author, pred_author, count in confused_pairs:
    print(f"True: {true_author}, Predicted: {pred_author}, Count: {count}")

# Save error analysis - Fixed version
error_analysis = {
    'error_rate': float(error_rate),  # Convert to float
    'total_errors': int(len(errors)),  # Convert to int
    'most_confused_pairs': [(int(t), int(p), int(c)) for t, p, c in confused_pairs],  # Convert all to int
    'per_author_accuracy': {str(k): float(v) for k, v in author_accuracies.items()}  # Convert values to float
}

with open(os.path.join(config.output_dir, f'{config.tokenization_method}_error_analysis.json'), 'w') as f:
    json.dump(error_analysis, f, indent=2)

# Create a comprehensive comparison script for all three methods
print("\nCreating comparison script...")

comparison_script = """
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load results for all three methods
methods = ['bpe', 'wordpiece', 'sentencepiece']
results = {}

for method in methods:
    with open(f'{method}_results.json', 'r') as f:
        results[method] = json.load(f)

# Create comparison DataFrame
comparison_data = []
for method, data in results.items():
    comparison_data.append({
        'Method': method.upper(),
        'Test Accuracy': data['test_results']['eval_accuracy'],
        'Test F1': data['test_results']['eval_f1'],
        'Test Top-5 Accuracy': data['test_results']['eval_top_5_accuracy'],
        'Model': data['model_name']
    })

df = pd.DataFrame(comparison_data)
print(df)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Accuracy comparison
axes[0].bar(df['Method'], df['Test Accuracy'])
axes[0].set_title('Test Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1)

# F1 Score comparison
axes[1].bar(df['Method'], df['Test F1'])
axes[1].set_title('Test F1 Score Comparison')
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0, 1)

# Top-5 Accuracy comparison
axes[2].bar(df['Method'], df['Test Top-5 Accuracy'])
axes[2].set_title('Test Top-5 Accuracy Comparison')
axes[2].set_ylabel('Top-5 Accuracy')
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('tokenization_methods_comparison.png', dpi=300)
plt.show()

# Save comparison table
df.to_csv('tokenization_methods_comparison.csv', index=False)
print("\\nComparison saved to tokenization_methods_comparison.csv")
"""

with open(os.path.join(config.output_dir, 'compare_methods.py'), 'w') as f:
    f.write(comparison_script)

# Print final summary
print("\n" + "="*50)
print(f"TRAINING COMPLETE - {config.tokenization_method.upper()}")
print("="*50)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
print(f"Test Top-5 Accuracy: {test_results['eval_top_5_accuracy']:.4f}")
print(f"Model saved to: {os.path.join(config.output_dir, f'{config.tokenization_method}_model')}")
print("="*50)
