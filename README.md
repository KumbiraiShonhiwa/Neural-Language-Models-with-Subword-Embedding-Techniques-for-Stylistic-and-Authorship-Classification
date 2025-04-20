

# Neural Language Models with Subword Embedding Techniques for Stylistic and Authorship Classification

**Authors:** Kumbirai Shonhiwa, Thando Dlamini, Given Chauke  
**Date:** April 2025

## 🧠 Project Overview

This project explores the use of **neural language models** combined with **subword embedding techniques** to improve **stylistic classification** and **authorship attribution**, particularly in **African languages**. By leveraging state-of-the-art transformer models and advanced tokenization strategies, we aim to overcome the limitations of traditional NLP methods in low-resource, morphologically rich language contexts.

---

## 📚 Background

Traditional NLP techniques—like Bag-of-Words, TF-IDF, or even word2vec—struggle with:
- Out-of-vocabulary terms
- Language-specific morphology
- Stylistic nuances in authorial voice

Subword-based approaches like **Byte-Pair Encoding (BPE)**, **WordPiece**, and **SentencePiece** address these issues by breaking words into smaller, meaningful units. When used with **transformer models** such as BERT and XLM-R, these techniques enhance both **semantic** and **syntactic** understanding, offering improved performance in nuanced NLP tasks.

---

## 📦 Dataset: AfriSenti

We use the **AfriSenti dataset**, the largest publicly available African language sentiment analysis corpus:

- ✅ **110,000+ tweets** across **14 African languages**
- ✅ Annotated for **sentiment** (positive, negative, neutral)
- ✅ Released as part of **SemEval 2023 Task 12**
- ✅ Anonymized and licensed under **CC BY 4.0**
- ✅ Accessible via HuggingFace's `datasets` library

---

## 🔧 Methodology

### 🔠 Subword Tokenization Techniques

We apply and compare several subword-aware tokenization methods:
- **N-grams**: Helps capture meaningful chunks in morphologically complex text
- **Byte Pair Encoding (BPE)**: Reduces token sparsity
- **SentencePiece**: Effective for languages with inconsistent spacing
- **WordPiece**: Robust handling of rare and compound words

### 🔁 Transfer Learning

We fine-tune pre-trained multilingual transformer models:
- **mBERT (Multilingual BERT)**
- **XLM-R (Cross-lingual RoBERTa)**

These models offer foundational language understanding and can be adapted for African language nuances with subword tokenization.

---

## 📏 Evaluation Metrics

We evaluate our models using:

- **Accuracy** – Overall correctness
- **Precision & Recall** – Quality and completeness of predictions
- **F1-Score** – Harmonic mean of precision and recall
- **Robustness Metrics**:
  - **Dialect & Domain Generalization**
  - **Noise Resilience** (handling misspellings, informal structures)

---

## 🚀 Expected Contributions

- 🧩 A subword tokenization framework tailored for **African languages**
- 📈 Improved **authorship attribution** and **stylistic analysis** performance
- 🔍 Enhanced model **generalization across dialects**
- 🧵 Reduced dependence on large annotated datasets
- 📚 Contribution to **cultural heritage preservation** through language technology

---

## 📁 Project Structure

```bash
├── data/                   # Preprocessed datasets (AfriSenti)
├── models/                 # Fine-tuned transformer models
├── tokenizers/             # Subword tokenization scripts
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Core training and evaluation scripts
├── results/                # Evaluation outputs
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 💡 Future Work

- Extend to authorship detection in **long-form literature**
- Develop a multilingual, subword-based authorship benchmark
- Investigate **zero-shot** and **few-shot** learning for low-resource languages

---

## 📜 License

This project is licensed under the **Creative Commons Attribution 4.0 International License**.  
Feel free to use, modify, and distribute for academic and non-commercial purposes.

---
