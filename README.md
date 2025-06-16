

# Neural Language Models with Subword Embedding Techniques for Stylistic and Authorship Classification

**Authors:** Kumbirai Shonhiwa, Thando Dlamini, Given Chauke  
**Date:** April 2025

## Project Overview

This project explores the use of **neural language models** combined with **subword embedding techniques** to improve **stylistic classification** and **authorship attribution**, particularly in **African languages**. By leveraging state-of-the-art transformer models and advanced tokenization strategies, we aim to overcome the limitations of traditional NLP methods in low-resource, morphologically rich language contexts.

---

## Background

Traditional NLP techniques—like Bag-of-Words, TF-IDF, or even word2vec—struggle with:
- Out-of-vocabulary terms
- Language-specific morphology
- Stylistic nuances in authorial voice

Subword-based approaches like **Byte-Pair Encoding (BPE)**, **WordPiece**, and **SentencePiece** address these issues by breaking words into smaller, meaningful units. When used with **transformer models** such as BERT and XLM-R, these techniques enhance both **semantic** and **syntactic** understanding, offering improved performance in nuanced NLP tasks.

---

## Dataset

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

## Expected Contributions

- Improved **authorship attribution** and **stylistic analysis** performance
- Enhanced model **generalization across dialects**
