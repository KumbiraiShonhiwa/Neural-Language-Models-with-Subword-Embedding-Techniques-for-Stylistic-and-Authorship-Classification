# Twitter Authorship Attribution Dataset

This dataset has been preprocessed for authorship attribution using three different subword tokenization methods.

## Dataset Overview

- **Source**: Twitter data from multiple users

## Preprocessing Steps

1. **Text Cleaning**:
   - Normalized byte string format
   - Replaced URLs with [URL] token
   - Replaced user mentions with [USER] token
   - Replaced hashtags with [HASHTAG] token
   - Removed extra spaces and newlines

2. **Dataset Balancing**:
   - Filtered to include only authors with at least {min_tweets} tweets
   - Limited to {max_authors} authors with the most tweets
   - Balanced by sampling equal number of tweets per author

3. **Tokenization**:
   - Applied three different subword tokenization methods:
     - Byte-Pair Encoding (BPE) using XLM-RoBERTa Base
     - WordPiece using mBERT
     - SentencePiece using XLM-RoBERTa Large

4. **Stylometric Analysis**:
   - Analyzed common patterns across the dataset

## File Descriptions

For each tokenization method, you'll find three datasets:
- `bpe_train_dataset.pt`, `bpe_val_dataset.pt`, `bpe_test_dataset.pt`
- `wordpiece_train_dataset.pt`, `wordpiece_val_dataset.pt`, `wordpiece_test_dataset.pt`
- `sentencepiece_train_dataset.pt`, `sentencepiece_val_dataset.pt`, `sentencepiece_test_dataset.pt`

Original processed data is also available in CSV format:
- `train.csv`, `val.csv`, `test.csv`

## Usage

```python
import torch

# Load the dataset
train_dataset = torch.load('bpe_train_dataset.pt')

# Each dataset contains:
# - input_ids: tokenized and encoded text
# - attention_mask: padding masks
# - labels: numeric author IDs
# - author_map: dictionary mapping original author IDs to numeric IDs

# Example usage with HuggingFace Trainer:
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",  # Use matching model for tokenization method
    num_labels=len(train_dataset['author_map'])
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

