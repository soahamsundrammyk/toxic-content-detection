# Toxic Content Detection with Explainability — Design Document

**Date**: 2026-04-02
**Author**: Soaham Sundram
**Status**: Approved

## Goal

Build a multilabel toxicity classifier on the Jigsaw dataset (160K comments, 6 categories) that progresses through 3 model architectures with a final explainability layer. This is a portfolio/resume project designed to demonstrate ML fundamentals, deep learning, transformer fine-tuning, and model interpretability.

## Dataset

- **Source**: Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Size**: ~160K Wikipedia comments
- **Labels**: 6 binary labels (multilabel): `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Key challenge**: Heavy class imbalance — `threat` (~0.3%) and `identity_hate` (~0.9%) are very rare

## Architecture

```
toxic-content-detection/
├── data/                          ← Kaggle dataset (gitignored)
│   └── train.csv
├── notebooks/
│   ├── 01_exploration_baseline.ipynb   ← Phase 1: EDA + TF-IDF + LogReg
│   ├── 02_bilstm_glove.ipynb           ← Phase 2: BiLSTM + GloVe embeddings
│   ├── 03_roberta_finetuning.ipynb     ← Phase 3: Fine-tune RoBERTa
│   └── 04_explainability.ipynb         ← Phase 4: LIME/SHAP + attention viz
├── src/
│   ├── __init__.py
│   ├── dataset.py       ← Load CSV, clean text, train/val/test split, PyTorch Dataset classes
│   ├── metrics.py       ← F1, ROC-AUC, confusion matrices, comparison tables
│   ├── models.py        ← BiLSTM class, RoBERTa wrapper
│   ├── training.py      ← Training loop, focal loss, early stopping
│   └── explain.py       ← LIME/SHAP wrappers, attention extraction
├── models/              ← Saved model checkpoints (gitignored)
├── results/             ← Charts, metrics JSON, comparison tables
├── requirements.txt
├── .gitignore
└── README.md            ← Project summary with results (written last)
```

## Phase Progression

### Phase 1: Data Exploration + TF-IDF Baseline
- **Model**: TF-IDF vectorization + Logistic Regression (OneVsRest)
- **Purpose**: Understand the data, establish baseline metrics
- **Imbalance handling**: `class_weight='balanced'`
- **Builds**: `src/dataset.py`, `src/metrics.py`, notebook 01
- **Concepts taught**: pandas, matplotlib, TF-IDF, logistic regression, precision/recall/F1, ROC-AUC, stratified splitting

### Phase 2: BiLSTM + GloVe Embeddings
- **Model**: Bidirectional LSTM with pre-trained GloVe 6B 100d embeddings
- **Purpose**: Introduce deep learning, sequence modeling, PyTorch fundamentals
- **Imbalance handling**: Focal loss
- **Builds**: `src/models.py` (BiLSTM), `src/training.py`, notebook 02
- **Concepts taught**: word embeddings, RNNs, LSTM gates, bidirectionality, PyTorch tensors/Dataset/DataLoader, training loops, focal loss, early stopping

### Phase 3: Fine-tuned RoBERTa
- **Model**: `roberta-base` from HuggingFace with classification head
- **Purpose**: Transformer fine-tuning, tokenization, transfer learning
- **Imbalance handling**: Focal loss + weighted sampling
- **Memory strategy**: batch size 8, gradient accumulation (2 steps), fp16 if MPS supports it, fallback to Colab
- **Builds**: RoBERTa wrapper in `src/models.py`, notebook 03
- **Concepts taught**: self-attention, tokenization (BPE), transfer learning, learning rate scheduling, gradient accumulation

### Phase 4: Explainability
- **Tools**: LIME, SHAP, attention visualization
- **Purpose**: Interpret model predictions, identify bias, build trust
- **Builds**: `src/explain.py`, notebook 04
- **Concepts taught**: LIME (local surrogate models), SHAP (Shapley values), attention weight extraction, when to trust/distrust attention

## Data Pipeline

- **Split**: 80% train / 10% val / 10% test, stratified on multilabel combination
- **Test set**: Touched exactly once for final comparison table
- **Text preprocessing**: Minimal (lowercase, basic cleaning) — let models learn from raw text
- **Reproducibility**: `random_state=42` everywhere

## Evaluation Strategy

- **Per-label**: F1 score, ROC-AUC, confusion matrix
- **Aggregate**: Macro F1, Macro ROC-AUC
- **Comparison table**: All 3 models side-by-side, highlighting improvement trajectory
- **Special attention**: `threat` and `identity_hate` F1 (hardest due to rarity)

## Expected Resume Line

> Trained multilabel toxicity classifier on 160K comments progressing from TF-IDF baseline to BiLSTM+GloVe to fine-tuned RoBERTa; improved macro F1 from X to Y with Z% ROC-AUC across 6 categories; built LIME/SHAP explainability layer for token-level attribution

## Hardware

- MacBook Pro M3, 16GB RAM
- PyTorch MPS backend for GPU acceleration
- Fallback: Google Colab free tier for Phase 3 if memory is tight

## Out of Scope

- Web app / API / deployment
- MLflow / Weights & Biases experiment tracking
- Hyperparameter search (grid/random/Bayesian)
- Docker / containerization
- Training on test labels (Jigsaw test set has no public labels)

## Learning Approach

Each phase follows: Concept explanation → Questions until understood → Code with line-by-line explanation → Run & observe → Learning journal (3-5 bullet points) → Interview mock questions (2-3 per phase)
