

# Neural Language Models with Subword Embedding Techniques for Stylistic and Authorship Classification

## Project Overview

This project explores the use of **neural language models** combined with **subword embedding techniques** to improve **stylistic classification** and **authorship attribution**. By leveraging state-of-the-art transformer models and advanced tokenization strategies.

**semantic** and **syntactic** understanding, offering improved performance in nuanced NLP tasks.

---

## Dataset

The final dataset maintains precise balance with exactly 200 tweets per author, distributed across training (14,000 tweets, 70\%), validation (3,000 tweets, 15\%), and test (3,000 tweets, 15\%) sets using stratified sampling to ensure proportional author
representation in each split.

## Methodology

### Subword Tokenization Techniques

We apply and compare several subword-aware tokenization methods:
- **Byte Pair Encoding (BPE)**: Reduces token sparsity
- **SentencePiece**: Effective for languages with inconsistent spacing
- **WordPiece**: Robust handling of rare and compound words

### Transfer Learning

We fine-tune pre-trained multilingual transformer models:
- **mBERT (Multilingual BERT)**
- **XLM-R (Cross-lingual RoBERTa)**

These models offer foundational language understanding and can be adapted for African language nuances with subword tokenization.

---

## Evaluation Metrics

We evaluate our models using:

- **Accuracy** – Overall correctness
- **Precision & Recall** – Quality and completeness of predictions
- **F1-Score** – Harmonic mean of precision and recall

---
