

# Neural Language Models with Subword Embedding Techniques for Stylistic and Authorship Classification

## Project Overview

This project explores the use of **neural language models** combined with **subword embedding techniques** to improve **stylistic classification** and **authorship attribution**. By leveraging state-of-the-art transformer models and advanced tokenization strategies.

---

## Contents of the Zip File
Root Directory

README.md - This file, containing project documentation
requirements.txt - Python package dependencies

* /src/ - Source Code

* /data/ - Data Directory

* /docs/ - Documentation

Contains presentation slides and additional documentation

* /notebooks/ - Jupyter Notebooks

* /results/ - Output Directory

Setup Instructions
Prerequisites

Python 3.8 or higher
Google Colab account  or local GPU environment
Google Drive account for data storage

Installation

1. Google Colab 

Upload the project files to your Google Colab environment
Install dependencies:

```
python!pip install -r requirements.txt

```

2. Local Environment

Clone or extract the project files
Create a virtual environment:

```

bashpython -m venv nlm_env
source nlm_env/bin/activate  # On Windows: nlm_env\Scripts\activate

```

Install dependencies:

```
bashpip install -r requirements.txt
```
---

## Background

Traditional NLP techniques—like Bag-of-Words, TF-IDF, or even word2vec—struggle with:
- Out-of-vocabulary terms
- Language-specific morphology
- Stylistic nuances in authorial voice

Subword-based approaches like **Byte-Pair Encoding (BPE)**, **WordPiece**, and **SentencePiece** address these issues by breaking words into smaller, meaningful units. When used with **transformer models** such as BERT and XLM-R, these techniques enhance both **semantic** and **syntactic** understanding, offering improved performance in nuanced NLP tasks.

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

## Expected Contributions

- Improved **authorship attribution** and **stylistic analysis** performance
- Enhanced model **generalization across dialects**
