# Toxic Content Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a multilabel toxicity classifier progressing through 3 architectures (TF-IDF → BiLSTM → RoBERTa) with LIME/SHAP explainability, designed as a portfolio project with deep learning understanding.

**Architecture:** 4 notebooks for exploration/storytelling + 5 Python modules in `src/` for reusable logic. Each phase builds on the previous, with shared data pipeline and evaluation code. The learner (Soaham) has basic Python knowledge and no ML project experience — every concept must be explained inline before coding it.

**Tech Stack:** Python, pandas, scikit-learn, PyTorch (MPS backend), HuggingFace Transformers, LIME, SHAP, matplotlib/seaborn

**Hardware:** MacBook Pro M3, 16GB RAM

---

## Task 0: Project Setup

**Files:**
- Create: `src/__init__.py`
- Create: `.gitignore`
- Modify: `requirements.txt`
- Delete: `notebooks/01_data_exploration.ipynb` (replaced by new version)

**Step 1: Initialize git repo**

```bash
cd /Users/soahamsundram/Documents/GitHub/toxic-content-detection
git init
```

**Step 2: Create .gitignore**

```gitignore
# Data (too large for git)
data/*.csv
data/*.zip

# Model checkpoints (too large)
models/*.pt
models/*.bin

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store

# Virtual environment
venv/
.venv/
```

**Step 3: Create virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 4: Create src/__init__.py**

```python
# Empty file — makes src/ a Python package so notebooks can import from it
```

**Step 5: Download dataset**

Manual: Go to https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
Download `train.csv.zip`, unzip, place `train.csv` in `data/`.

Or via CLI:
```bash
pip install kaggle
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data/
unzip data/jigsaw-toxic-comment-classification-challenge.zip -d data/
```

**Step 6: Verify data exists**

```bash
head -2 data/train.csv
wc -l data/train.csv
```
Expected: ~159572 lines, columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

**Step 7: Delete old notebook**

```bash
rm notebooks/01_data_exploration.ipynb
```

**Step 8: Commit**

```bash
git add .gitignore src/__init__.py requirements.txt docs/
git commit -m "chore: project setup with gitignore, src package, and design docs"
```

---

## Task 1: Build `src/dataset.py` — Data Loading & Splitting

**Files:**
- Create: `src/dataset.py`

**Concepts to teach before coding:**
- What a Python module is (a .py file you can import)
- `pandas.read_csv()` — loading tabular data
- Why we split data into train/val/test (analogy: studying, practice exam, final exam)
- Stratified splitting — ensuring rare labels appear in every split
- What `random_state` does (reproducibility)
- Type hints — what they are and why they help

**Step 1: Write `src/dataset.py`**

```python
"""
Dataset loading and splitting for Jigsaw Toxic Comment Classification.

This module handles:
1. Loading the CSV data
2. Basic text cleaning
3. Stratified train/val/test splitting
4. (Later) PyTorch Dataset classes for deep learning phases
"""

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# The 6 toxicity labels in the Jigsaw dataset
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load the Jigsaw training CSV and do basic validation.

    Args:
        data_dir: Path to the directory containing train.csv

    Returns:
        DataFrame with columns: id, comment_text, + 6 label columns
    """
    path = Path(data_dir) / 'train.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {path}. "
            "Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"
        )

    df = pd.read_csv(path)

    # Fill any missing comments with empty string
    df['comment_text'] = df['comment_text'].fillna('')

    return df


def clean_text(text: str) -> str:
    """Minimal text cleaning — lowercase and normalize whitespace.

    We keep it minimal on purpose: let the models learn from (mostly) raw text.
    Heavy cleaning can remove signal (e.g., ALL CAPS often indicates anger).

    Args:
        text: Raw comment text

    Returns:
        Cleaned text
    """
    # Lowercase
    text = text.lower()
    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_splits(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with stratified sampling.

    Stratification ensures each split has roughly the same proportion
    of each label combination. This is critical when labels like 'threat'
    are only ~0.3% of the data.

    Args:
        df: Full dataset DataFrame
        test_size: Fraction for test set (default 0.1 = 10%)
        val_size: Fraction for validation set (default 0.1 = 10%)
        random_state: Seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Create a stratification key by joining all 6 labels into a string
    # e.g., a comment that is toxic + obscene becomes "100100"
    stratify_key = df[LABEL_COLS].astype(str).agg(''.join, axis=1)

    # Some label combinations might be extremely rare (only 1 sample).
    # Stratification fails if a group has fewer than 2 samples.
    # Replace rare combinations with a generic key.
    combo_counts = stratify_key.value_counts()
    rare_combos = combo_counts[combo_counts < 2].index
    stratify_key = stratify_key.replace(rare_combos, 'rare')

    # First split: separate out the test set
    temp_size = test_size + val_size  # e.g., 0.2
    train_df, temp_df, strat_train, strat_temp = train_test_split(
        df, stratify_key, test_size=temp_size, random_state=random_state, stratify=stratify_key
    )

    # Second split: divide remaining into val and test
    # val_size / (val_size + test_size) gives the right proportion
    val_fraction = val_size / temp_size  # e.g., 0.5

    # Rebuild stratify key for the temp subset
    strat_temp_key = strat_temp.reset_index(drop=True)
    temp_df = temp_df.reset_index(drop=True)

    # Handle rare combos in temp set too
    temp_combo_counts = strat_temp_key.value_counts()
    rare_temp = temp_combo_counts[temp_combo_counts < 2].index
    strat_temp_key = strat_temp_key.replace(rare_temp, 'rare')

    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_fraction), random_state=random_state, stratify=strat_temp_key
    )

    # Reset all indices so they're clean
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print a summary of the data splits showing sizes and label distributions."""
    print(f"{'Split':<8} {'Size':>7} {'% of total':>10}")
    print("-" * 28)
    total = len(train_df) + len(val_df) + len(test_df)
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pct = len(split_df) / total * 100
        print(f"{name:<8} {len(split_df):>7,} {pct:>9.1f}%")

    print(f"\nLabel distribution (% positive) per split:")
    print(f"{'Label':<15} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 40)
    for label in LABEL_COLS:
        train_pct = train_df[label].mean() * 100
        val_pct = val_df[label].mean() * 100
        test_pct = test_df[label].mean() * 100
        print(f"{label:<15} {train_pct:>6.2f}% {val_pct:>6.2f}% {test_pct:>6.2f}%")
```

**Step 2: Verify the module imports**

```bash
cd /Users/soahamsundram/Documents/GitHub/toxic-content-detection
source venv/bin/activate
python -c "from src.dataset import load_data, get_splits, LABEL_COLS; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/dataset.py
git commit -m "feat: add dataset loading and stratified splitting module"
```

---

## Task 2: Build `src/metrics.py` — Evaluation Functions

**Files:**
- Create: `src/metrics.py`

**Concepts to teach before coding:**
- Precision: "Of everything you flagged, how much was actually bad?"
- Recall: "Of all actual bad stuff, how much did you catch?"
- F1: harmonic mean of precision and recall (balances both)
- ROC-AUC: how well the model separates classes across all thresholds
- Macro average: treat each label equally (important for rare labels)
- Confusion matrix: 2x2 grid showing correct vs incorrect predictions

**Step 1: Write `src/metrics.py`**

```python
"""
Evaluation metrics for multilabel toxicity classification.

Provides consistent evaluation across all model phases:
- Per-label F1, precision, recall, ROC-AUC
- Macro-averaged metrics
- Confusion matrices
- Model comparison tables
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset import LABEL_COLS


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: List[str] = LABEL_COLS,
) -> Dict:
    """Compute all evaluation metrics for multilabel predictions.

    Args:
        y_true: Ground truth labels, shape (n_samples, n_labels)
        y_pred: Binary predictions, shape (n_samples, n_labels)
        y_proba: Predicted probabilities, shape (n_samples, n_labels).
                 Optional — ROC-AUC requires this.
        label_names: Names of the labels

    Returns:
        Dictionary with per-label and macro metrics
    """
    results = {
        'per_label': {},
        'macro': {},
    }

    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []

    for i, label in enumerate(label_names):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)

        label_metrics = {
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'support': int(y_true[:, i].sum()),
        }

        if y_proba is not None:
            auc = roc_auc_score(y_true[:, i], y_proba[:, i])
            label_metrics['roc_auc'] = round(auc, 4)
            auc_scores.append(auc)

        results['per_label'][label] = label_metrics
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)

    results['macro'] = {
        'f1': round(np.mean(f1_scores), 4),
        'precision': round(np.mean(precision_scores), 4),
        'recall': round(np.mean(recall_scores), 4),
    }

    if auc_scores:
        results['macro']['roc_auc'] = round(np.mean(auc_scores), 4)

    return results


def print_metrics(results: Dict, model_name: str = "Model") -> None:
    """Pretty-print evaluation metrics as a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 70}")

    header = f"{'Label':>15s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s}"
    has_auc = 'roc_auc' in list(results['per_label'].values())[0]
    if has_auc:
        header += f" | {'AUC':>6s}"
    header += f" | {'Support':>7s}"

    print(header)
    print("-" * 70)

    for label, metrics in results['per_label'].items():
        line = f"{label:>15s} | {metrics['f1']:6.4f} | {metrics['precision']:6.4f} | {metrics['recall']:6.4f}"
        if has_auc:
            line += f" | {metrics['roc_auc']:6.4f}"
        line += f" | {metrics['support']:>7d}"
        print(line)

    print("-" * 70)
    macro = results['macro']
    line = f"{'MACRO AVG':>15s} | {macro['f1']:6.4f} | {macro['precision']:6.4f} | {macro['recall']:6.4f}"
    if 'roc_auc' in macro:
        line += f" | {macro['roc_auc']:6.4f}"
    line += f" |"
    print(line)
    print(f"{'=' * 70}\n")


def save_results(results: Dict, model_name: str, results_dir: str = 'results') -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Output from evaluate_predictions()
        model_name: Name like 'tfidf_logreg', 'bilstm_glove', 'roberta'
        results_dir: Directory to save to
    """
    path = Path(results_dir)
    path.mkdir(exist_ok=True)

    output = {'model': model_name, **results}
    filepath = path / f"{model_name}_results.json"

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {filepath}")


def load_all_results(results_dir: str = 'results') -> List[Dict]:
    """Load all saved result JSON files for comparison."""
    path = Path(results_dir)
    results = []
    for filepath in sorted(path.glob('*_results.json')):
        with open(filepath) as f:
            results.append(json.load(f))
    return results


def print_comparison_table(results_list: List[Dict]) -> None:
    """Print a side-by-side comparison of all models.

    This produces the final comparison table for the resume.
    """
    if not results_list:
        print("No results to compare.")
        return

    print(f"\n{'=' * 80}")
    print(f"  MODEL COMPARISON")
    print(f"{'=' * 80}")

    # Header
    header = f"{'Model':<25s} | {'Macro F1':>8s} | {'Macro AUC':>9s}"
    for label in ['threat', 'identity_hate']:
        header += f" | {label + ' F1':>15s}"
    print(header)
    print("-" * 80)

    for result in results_list:
        name = result['model']
        macro_f1 = result['macro']['f1']
        macro_auc = result['macro'].get('roc_auc', 'N/A')

        line = f"{name:<25s} | {macro_f1:>8.4f} | "
        if isinstance(macro_auc, float):
            line += f"{macro_auc:>9.4f}"
        else:
            line += f"{'N/A':>9s}"

        for label in ['threat', 'identity_hate']:
            label_f1 = result['per_label'][label]['f1']
            line += f" | {label_f1:>15.4f}"

        print(line)

    print(f"{'=' * 80}\n")
```

**Step 2: Verify**

```bash
python -c "from src.metrics import evaluate_predictions, print_metrics; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/metrics.py
git commit -m "feat: add evaluation metrics module with comparison tables"
```

---

## Task 3: Notebook 01 — Data Exploration + TF-IDF Baseline

**Files:**
- Create: `notebooks/01_exploration_baseline.ipynb`

**Concepts to teach (in notebook markdown cells):**
- How pandas DataFrames work (rows/columns, indexing, filtering)
- What `.shape`, `.head()`, `.describe()`, `.value_counts()` do
- matplotlib/seaborn basics (bar chart, histogram, heatmap)
- Class imbalance — why accuracy is misleading
- TF-IDF — plain English + the actual math (TF * IDF)
- Logistic Regression — learns a weight per word, outputs probability
- OneVsRest — train one classifier per label for multilabel
- `class_weight='balanced'` — sklearn auto-adjusts for rare labels
- `fit_transform` vs `transform` — why this prevents data leakage

**Notebook structure (each section = markdown explanation → code → observe output):**

1. **Imports & Setup** — explain what each library does
2. **Load Data** — use `src.dataset.load_data()`, explore with pandas
3. **Label Distribution** — bar chart + counts, discuss imbalance
4. **Label Correlation** — heatmap, discuss co-occurrence patterns
5. **Text Length Analysis** — histogram, compare toxic vs clean
6. **Train/Val/Test Split** — use `src.dataset.get_splits()`, verify distributions
7. **TF-IDF Vectorization** — explain parameters, fit on train only
8. **Train Logistic Regression** — OneVsRest with balanced weights
9. **Evaluate on Val Set** — use `src.metrics`, print tables, plot bar charts
10. **Top Predictive Words** — inspect model weights for interpretability
11. **Save Baseline Results** — JSON for later comparison
12. **Learning Journal** — empty cells for learner to fill in
13. **Interview Prep Questions** — 5 questions with space for answers

**Step 1: Write the notebook**

The notebook will be written as a complete .ipynb file. Key code cells use `src.dataset` and `src.metrics` imports (showing the learner how modules work).

All notebook code should add `sys.path` setup at the top:
```python
import sys
sys.path.insert(0, '..')
```

**Step 2: Run notebook end-to-end**

```bash
cd notebooks
jupyter nbconvert --execute 01_exploration_baseline.ipynb --to notebook --inplace
```

**Step 3: Commit**

```bash
git add notebooks/01_exploration_baseline.ipynb results/
git commit -m "feat: Phase 1 complete — data exploration and TF-IDF baseline"
```

---

## Task 4: Build `src/models.py` — BiLSTM Model

**Files:**
- Create: `src/models.py`

**Concepts to teach before coding:**
- What a tensor is (a multi-dimensional array, like numpy but GPU-aware)
- Word embeddings: each word → a vector of numbers that captures meaning
  - "king" and "queen" are close in embedding space
  - GloVe: pre-trained on billions of words, we just load them
- RNN: processes words one at a time, passes a "memory" forward
  - Problem: forgets early words in long sequences ("vanishing gradient")
- LSTM: special RNN with gates that control what to remember/forget
  - Forget gate: "should I keep this memory?"
  - Input gate: "should I store this new information?"
  - Output gate: "what should I pass forward?"
- BiLSTM: two LSTMs — one reads left→right, one reads right→left
  - "Not bad" — the forward LSTM sees "not" before "bad", backward sees "bad" before "not"
  - Concatenate both outputs for full context
- `nn.Module`: PyTorch's base class for all neural networks
- `forward()`: defines what happens when data passes through the model

**Step 1: Write BiLSTM model class in `src/models.py`**

```python
"""
Neural network model definitions for toxicity classification.

Phase 2: BiLSTM with GloVe embeddings
Phase 3: RoBERTa fine-tuning wrapper (added later)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for multilabel text classification.

    Architecture:
        text → embedding → BiLSTM → dropout → linear → sigmoid

    The embedding layer can be initialized with pre-trained GloVe vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_labels: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Args:
            vocab_size: Number of unique words in our vocabulary
            embedding_dim: Size of each word vector (100 for GloVe 6B 100d)
            hidden_dim: Number of features in the LSTM hidden state
            num_labels: Number of output labels (6 for Jigsaw)
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate for regularization
            pretrained_embeddings: Optional numpy array of GloVe vectors
        """
        super().__init__()

        # Embedding layer: converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # If we have pre-trained GloVe vectors, use them
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Don't update GloVe vectors during training (freeze them)
            self.embedding.weight.requires_grad = False

        # BiLSTM: processes the sequence in both directions
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq_len, features)
            bidirectional=True,     # read forward AND backward
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout: randomly zeros some values during training (prevents overfitting)
        self.dropout = nn.Dropout(dropout)

        # Linear layer: maps LSTM output to 6 label predictions
        # hidden_dim * 2 because bidirectional concatenates forward + backward
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: text indices → toxicity probabilities.

        Args:
            x: Token indices, shape (batch_size, seq_length)

        Returns:
            Logits (raw scores before sigmoid), shape (batch_size, num_labels)
        """
        # x shape: (batch_size, seq_length) — integers like [4, 127, 8903, 2, ...]

        # Step 1: Look up embeddings for each word
        embedded = self.embedding(x)
        # shape: (batch_size, seq_length, embedding_dim)

        # Step 2: Pass through BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_length, hidden_dim * 2)
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)

        # Step 3: Take the last hidden states from both directions
        # Forward LSTM's last hidden state
        forward_hidden = hidden[-2]   # shape: (batch_size, hidden_dim)
        # Backward LSTM's last hidden state
        backward_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)
        # Concatenate them
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        # shape: (batch_size, hidden_dim * 2)

        # Step 4: Dropout + linear layer
        dropped = self.dropout(combined)
        logits = self.fc(dropped)
        # shape: (batch_size, num_labels)

        # Note: we return LOGITS, not probabilities.
        # The loss function (BCEWithLogitsLoss / FocalLoss) applies sigmoid internally.
        # For predictions, apply torch.sigmoid(logits) separately.
        return logits


def load_glove_embeddings(
    glove_path: str,
    word_to_idx: dict,
    embedding_dim: int = 100,
) -> np.ndarray:
    """Load pre-trained GloVe vectors and align them with our vocabulary.

    GloVe file format: each line is "word 0.123 -0.456 0.789 ..."
    We only keep vectors for words that appear in our dataset's vocabulary.

    Args:
        glove_path: Path to glove.6B.100d.txt
        word_to_idx: Dictionary mapping word → index in our vocabulary
        embedding_dim: Dimension of GloVe vectors (100 for 6B.100d)

    Returns:
        Numpy array of shape (vocab_size, embedding_dim)
    """
    vocab_size = len(word_to_idx)
    # Initialize with random small values (for words not in GloVe)
    embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    # Padding token (index 0) should be all zeros
    embeddings[0] = np.zeros(embedding_dim)

    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word_to_idx:
                idx = word_to_idx[word]
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[idx] = vector
                found += 1

    print(f"GloVe: found {found}/{vocab_size} words ({found/vocab_size*100:.1f}%)")
    return embeddings.astype(np.float32)
```

**Step 2: Verify**

```bash
python -c "
from src.models import BiLSTMClassifier
import torch
model = BiLSTMClassifier(vocab_size=1000, embedding_dim=100)
x = torch.randint(0, 1000, (4, 50))  # batch of 4, sequence length 50
out = model(x)
print(f'Output shape: {out.shape}')  # should be (4, 6)
print('OK')
"
```
Expected: `Output shape: torch.Size([4, 6])` then `OK`

**Step 3: Commit**

```bash
git add src/models.py
git commit -m "feat: add BiLSTM classifier with GloVe embedding support"
```

---

## Task 5: Build `src/training.py` — Training Loop + Focal Loss

**Files:**
- Create: `src/training.py`

**Concepts to teach before coding:**
- What a training loop is (show data → compute error → adjust weights → repeat)
- Loss function: measures how wrong the model is
  - BCE (Binary Cross-Entropy): standard loss for binary classification
  - Focal Loss: BCE but pays MORE attention to hard/rare examples
    - Analogy: a teacher spending extra time on the questions students keep getting wrong
- Optimizer: how we adjust weights (Adam — adaptive learning rate per parameter)
- Learning rate: how big the weight adjustments are (too big → overshoots, too small → too slow)
- Epoch: one full pass through the training data
- Batch: a small chunk of data processed at once (can't fit all 160K in memory)
- Early stopping: stop training when val performance stops improving (prevents overfitting)
- MPS device: using the M3 GPU for faster math

**Step 1: Write `src/training.py`**

```python
"""
Training loop, loss functions, and utilities for PyTorch models.

Includes:
- Focal Loss (handles class imbalance)
- Generic training loop with early stopping
- MPS/CPU device selection for M3 Mac
"""

import time
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Get the best available device (MPS for M3 Mac, else CPU).

    MPS = Metal Performance Shaders — Apple's GPU acceleration for PyTorch.
    """
    if torch.backends.mps.is_available():
        print("Using MPS (Apple M3 GPU)")
        return torch.device('mps')
    print("Using CPU")
    return torch.device('cpu')


class FocalLoss(nn.Module):
    """Focal Loss for multilabel classification with class imbalance.

    Standard BCE treats all examples equally. Focal Loss adds a factor
    that DOWN-weights easy/common examples and UP-weights hard/rare ones.

    Formula: FL(p) = -alpha * (1 - p)^gamma * log(p)

    - When the model is confident and CORRECT (p close to 1):
      (1 - p)^gamma is tiny → loss is very small (don't waste effort here)
    - When the model is confident and WRONG (p close to 0):
      (1 - p)^gamma is large → loss is large (pay attention!)

    gamma=0 makes this identical to regular BCE.
    gamma=2 is a common default that works well in practice.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor (1.0 = no class weighting)
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   0 = standard BCE, 2 = common default.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output BEFORE sigmoid, shape (batch_size, num_labels)
            targets: Ground truth, shape (batch_size, num_labels), values 0 or 1
        """
        # BCE with logits is numerically stable (avoids log(0))
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Convert logits to probabilities for the focal weight
        probs = torch.sigmoid(logits)

        # p_t = probability of the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Final loss
        loss = self.alpha * focal_weight * bce_loss

        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch (one full pass through training data).

    Returns:
        Average training loss for this epoch
    """
    model.train()  # Put model in training mode (enables dropout)
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Move data to device (MPS/CPU)
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass: data → model → predictions
        logits = model(inputs)
        loss = loss_fn(logits, labels)

        # Backward pass: compute gradients (how to adjust each weight)
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients

        # Gradient clipping: prevent exploding gradients in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()  # Disable gradient computation (saves memory during evaluation)
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluate model on a dataset (validation or test).

    Returns:
        Dictionary with 'loss', 'predictions' (binary), and 'probabilities'
    """
    model.eval()  # Evaluation mode (disables dropout)
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(inputs)
        loss = loss_fn(logits, labels)

        probs = torch.sigmoid(logits)  # Convert logits to probabilities

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return {
        'loss': total_loss / n_batches,
        'probabilities': all_probs,
        'predictions': (all_probs >= 0.5).astype(int),
        'labels': all_labels,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 10,
    patience: int = 3,
    save_path: Optional[str] = None,
) -> Dict:
    """Full training loop with early stopping.

    Early stopping: if validation loss doesn't improve for `patience` epochs,
    stop training and restore the best model. This prevents overfitting
    (memorizing training data instead of learning general patterns).

    Args:
        model: The neural network
        train_loader: Training data batches
        val_loader: Validation data batches
        loss_fn: Loss function (FocalLoss)
        optimizer: Optimizer (Adam)
        device: MPS or CPU
        n_epochs: Maximum number of epochs
        patience: Stop after this many epochs without improvement
        save_path: Where to save the best model checkpoint

    Returns:
        Training history (losses per epoch)
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': []}
    best_state = None

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # Evaluate on validation set
        val_results = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_results['loss']

        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(
            f"Epoch {epoch+1:>2d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss: {best_val_loss:.4f})")

    return history
```

**Step 2: Verify**

```bash
python -c "
from src.training import FocalLoss, get_device
import torch
loss_fn = FocalLoss(gamma=2.0)
logits = torch.randn(4, 6)
targets = torch.randint(0, 2, (4, 6)).float()
loss = loss_fn(logits, targets)
print(f'Loss: {loss.item():.4f}')
device = get_device()
print('OK')
"
```
Expected: a loss value, device info, and `OK`

**Step 3: Commit**

```bash
git add src/training.py
git commit -m "feat: add training loop with focal loss and early stopping"
```

---

## Task 6: Add PyTorch Dataset to `src/dataset.py`

**Files:**
- Modify: `src/dataset.py` (add ToxicDataset class and vocabulary builder)

**Concepts to teach before coding:**
- PyTorch Dataset: a class that holds your data and serves one sample at a time
- PyTorch DataLoader: wraps a Dataset and serves batches, handles shuffling
- Vocabulary: mapping from words to integer indices ("hello" → 42)
- Padding: making all sequences the same length by adding zeros
- Tokenization: splitting text into words (simple whitespace split for now)

**Step 1: Add to `src/dataset.py`**

Append the following classes after the existing code:

```python
import torch
from torch.utils.data import Dataset
from collections import Counter


class Vocabulary:
    """Maps words to integer indices and back.

    Special tokens:
        <PAD> = 0  (used to fill short sequences to equal length)
        <UNK> = 1  (used for words not in our vocabulary)
    """

    def __init__(self, max_size: int = 50000):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.max_size = max_size

    def build(self, texts: pd.Series) -> 'Vocabulary':
        """Build vocabulary from a pandas Series of text.

        Counts word frequencies and keeps the top max_size words.
        Only call this on TRAINING data (not val/test) to prevent data leakage.
        """
        counter = Counter()
        for text in texts:
            words = clean_text(text).split()
            counter.update(words)

        # Keep only the most common words
        for word, count in counter.most_common(self.max_size - 2):  # -2 for PAD and UNK
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        print(f"Vocabulary built: {len(self.word_to_idx)} words")
        return self

    def encode(self, text: str, max_length: int = 200) -> list:
        """Convert text to a list of integer indices.

        Args:
            text: Input text
            max_length: Truncate/pad to this length

        Returns:
            List of integers, length = max_length
        """
        words = clean_text(text).split()

        # Convert words to indices (use <UNK>=1 for unknown words)
        indices = [self.word_to_idx.get(w, 1) for w in words]

        # Truncate if too long
        indices = indices[:max_length]

        # Pad with zeros if too short
        indices = indices + [0] * (max_length - len(indices))

        return indices

    def __len__(self) -> int:
        return len(self.word_to_idx)


class ToxicDataset(Dataset):
    """PyTorch Dataset for the Jigsaw toxic comment data.

    Serves one (text_indices, labels) pair at a time.
    The DataLoader wraps this to create batches.
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_length: int = 200):
        """
        Args:
            df: DataFrame with 'comment_text' and label columns
            vocab: Vocabulary object for encoding text
            max_length: Maximum sequence length
        """
        self.texts = df['comment_text'].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        """How many samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get one sample by index.

        Returns a dictionary (not a tuple) so it's clear what each value is.
        """
        text = self.texts[idx]
        indices = self.vocab.encode(text, self.max_length)

        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
```

**Step 2: Verify**

```bash
python -c "
from src.dataset import Vocabulary, ToxicDataset
import pandas as pd
import torch

# Create a tiny test dataset
df = pd.DataFrame({
    'comment_text': ['hello world', 'this is toxic garbage', 'nice comment here'],
    'toxic': [0, 1, 0], 'severe_toxic': [0, 0, 0], 'obscene': [0, 1, 0],
    'threat': [0, 0, 0], 'insult': [0, 1, 0], 'identity_hate': [0, 0, 0],
})

vocab = Vocabulary(max_size=100).build(df['comment_text'])
dataset = ToxicDataset(df, vocab, max_length=10)
sample = dataset[1]
print(f'input_ids shape: {sample[\"input_ids\"].shape}')
print(f'labels shape: {sample[\"labels\"].shape}')
print('OK')
"
```
Expected: shapes (10,) and (6,), then `OK`

**Step 3: Commit**

```bash
git add src/dataset.py
git commit -m "feat: add Vocabulary and ToxicDataset for PyTorch data pipeline"
```

---

## Task 7: Notebook 02 — BiLSTM + GloVe

**Files:**
- Create: `notebooks/02_bilstm_glove.ipynb`

**Prerequisites:**
- Download GloVe embeddings: `glove.6B.100d.txt` (822 MB)
  - From: https://nlp.stanford.edu/data/glove.6B.zip
  - Unzip and place in `data/glove.6B.100d.txt`

**Notebook structure:**

1. **Recap & motivation** — what TF-IDF can't do, why we need sequence models
2. **Concept: Word Embeddings** — explain GloVe, show similar words example
3. **Concept: RNNs → LSTMs → BiLSTMs** — diagrams in markdown, build intuition
4. **Build vocabulary** — use `Vocabulary` class, explore word counts
5. **Load GloVe embeddings** — use `load_glove_embeddings`, check coverage
6. **Create PyTorch Datasets and DataLoaders** — explain batching, padding, shuffling
7. **Concept: Focal Loss** — why `class_weight='balanced'` isn't enough, the math
8. **Initialize model** — create `BiLSTMClassifier`, print architecture
9. **Concept: Training loop** — forward pass, loss, backward pass, optimizer step
10. **Train** — use `train_model()`, watch losses per epoch
11. **Plot training curves** — train vs val loss over epochs
12. **Evaluate on val set** — use `src.metrics`, compare with baseline table
13. **Error analysis** — look at examples where BiLSTM beats baseline and vice versa
14. **Save results** — JSON for comparison
15. **Learning journal** — empty cells
16. **Interview prep** — 5 questions

**Step 1: Write the notebook**

**Step 2: Download GloVe if not present**

```bash
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip glove.6B.50d.txt glove.6B.200d.txt glove.6B.300d.txt
```

**Step 3: Run notebook**

```bash
cd notebooks
jupyter nbconvert --execute 02_bilstm_glove.ipynb --to notebook --inplace
```

**Step 4: Commit**

```bash
git add notebooks/02_bilstm_glove.ipynb results/ models/
git commit -m "feat: Phase 2 complete — BiLSTM with GloVe embeddings"
```

---

## Task 8: Add RoBERTa Wrapper to `src/models.py`

**Files:**
- Modify: `src/models.py` (add RoBERTaClassifier class)

**Concepts to teach before coding:**
- Transformers: self-attention lets every word "look at" every other word simultaneously
- BERT vs RoBERTa: RoBERTa is BERT trained longer with better hyperparameters
- Transfer learning: RoBERTa already "understands" English — we just teach it our specific task
- Tokenization (BPE): splits rare words into subwords ("unhappiness" → "un", "happiness")
- Classification head: one new linear layer on top of RoBERTa
- Freezing vs fine-tuning: which layers to update

**Step 1: Add RoBERTa class to `src/models.py`**

```python
from transformers import RobertaModel, RobertaTokenizer


class RoBERTaClassifier(nn.Module):
    """Fine-tuned RoBERTa for multilabel toxicity classification.

    Architecture:
        text → RoBERTa tokenizer → RoBERTa encoder → dropout → linear → sigmoid

    We take the [CLS] token output (a summary of the whole input)
    and pass it through a classification head.
    """

    def __init__(
        self,
        num_labels: int = 6,
        model_name: str = 'roberta-base',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tokenized input, shape (batch_size, seq_length)
            attention_mask: 1 for real tokens, 0 for padding, shape (batch_size, seq_length)

        Returns:
            Logits, shape (batch_size, num_labels)
        """
        # Get RoBERTa outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)

        # Classification head
        dropped = self.dropout(cls_output)
        logits = self.classifier(dropped)

        return logits
```

**Step 2: Add RoBERTa dataset class to `src/dataset.py`**

```python
from transformers import RobertaTokenizer


class ToxicDatasetRoBERTa(Dataset):
    """PyTorch Dataset that tokenizes text using RoBERTa's tokenizer."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: RobertaTokenizer,
        max_length: int = 256,
    ):
        self.texts = df['comment_text'].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # RoBERTa tokenizer handles everything: subword splitting, special tokens, padding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
```

**Step 3: Update training.py evaluate function for RoBERTa**

Add an updated evaluate function or modify the existing one to handle `attention_mask`:

```python
@torch.no_grad()
def evaluate_roberta(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluate RoBERTa model (handles attention_mask in batch)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return {
        'loss': total_loss / n_batches,
        'probabilities': all_probs,
        'predictions': (all_probs >= 0.5).astype(int),
        'labels': all_labels,
    }
```

**Step 4: Verify**

```bash
python -c "
from src.models import RoBERTaClassifier
import torch
model = RoBERTaClassifier(num_labels=6)
ids = torch.randint(0, 50265, (2, 32))
mask = torch.ones(2, 32, dtype=torch.long)
out = model(ids, mask)
print(f'Output shape: {out.shape}')
print('OK')
"
```
Expected: downloads roberta-base (~500MB first time), then `Output shape: torch.Size([2, 6])` and `OK`

**Step 5: Commit**

```bash
git add src/models.py src/dataset.py src/training.py
git commit -m "feat: add RoBERTa classifier, tokenized dataset, and evaluation"
```

---

## Task 9: Notebook 03 — RoBERTa Fine-tuning

**Files:**
- Create: `notebooks/03_roberta_finetuning.ipynb`

**Notebook structure:**

1. **Recap** — what BiLSTM can't do, why transformers are better
2. **Concept: Self-Attention** — plain English + visual diagram (every word attends to every word)
3. **Concept: BERT/RoBERTa** — pre-training (masked language model), why it "understands" English
4. **Concept: Transfer Learning** — analogy (learning piano helps with guitar)
5. **Concept: BPE Tokenization** — show how RoBERTa tokenizes example sentences
6. **Create RoBERTa tokenizer and datasets** — explore tokenized outputs
7. **Memory management** — explain batch size, gradient accumulation, why 16GB is tight
8. **Initialize model** — create `RoBERTaClassifier`, count parameters
9. **Training setup** — AdamW optimizer, linear warmup schedule, explain each choice
10. **Train** — custom training loop with gradient accumulation
11. **Plot training curves** — compare with BiLSTM
12. **Evaluate on val set** — full metrics
13. **Three-model comparison table** — the resume moment
14. **Error analysis** — what RoBERTa gets right that BiLSTM doesn't
15. **Save results**
16. **Learning journal**
17. **Interview prep** — 5 questions

**Key memory management code for M3 16GB:**
```python
# Gradient accumulation: simulate larger batch by accumulating over multiple small batches
# effective_batch_size = batch_size * accumulation_steps = 8 * 2 = 16
batch_size = 8
accumulation_steps = 2
max_length = 256  # Not 512 — saves significant memory
```

**Step 1: Write the notebook**

**Step 2: Run notebook** (this will take ~30-60 min on M3)

**Step 3: Commit**

```bash
git add notebooks/03_roberta_finetuning.ipynb results/ models/
git commit -m "feat: Phase 3 complete — fine-tuned RoBERTa with gradient accumulation"
```

---

## Task 10: Build `src/explain.py` — Explainability Module

**Files:**
- Create: `src/explain.py`

**Concepts to teach before coding:**
- Why explainability matters (trust, debugging, bias detection, legal requirements)
- LIME: "What happens if I randomly remove words? Which removals change the prediction most?"
  - Local: explains ONE prediction, not the whole model
  - Model-agnostic: works on ANY model (LogReg, LSTM, BERT)
- SHAP: Uses game theory (Shapley values) to fairly distribute "credit" among words
  - Each word gets a score: how much did it contribute to the prediction?
- Attention visualization: which tokens did the transformer focus on?
  - Caveat: attention ≠ explanation (active research debate — good interview topic!)

**Step 1: Write `src/explain.py`**

```python
"""
Explainability tools for toxicity classification.

Provides:
- LIME explanations (model-agnostic, local)
- SHAP explanations (Shapley value-based)
- Attention visualization for transformer models
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.dataset import LABEL_COLS


def lime_explain(
    text: str,
    predict_fn: Callable,
    label_names: List[str] = LABEL_COLS,
    num_features: int = 10,
    num_samples: int = 500,
) -> 'lime.explanation.Explanation':
    """Generate a LIME explanation for a single comment.

    LIME works by:
    1. Taking the original text
    2. Creating many perturbed versions (randomly removing words)
    3. Getting predictions for all perturbed versions
    4. Fitting a simple linear model to see which words matter most

    Args:
        text: The comment to explain
        predict_fn: Function that takes a list of strings → array of probabilities
        label_names: Names of the labels
        num_features: How many top words to show
        num_samples: How many perturbations to try (more = slower but more accurate)

    Returns:
        LIME Explanation object (can be visualized)
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=label_names)

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=list(range(len(label_names))),
    )

    return explanation


def shap_explain(
    texts: List[str],
    predict_fn: Callable,
    label_names: List[str] = LABEL_COLS,
) -> 'shap.Explanation':
    """Generate SHAP explanations for a batch of comments.

    SHAP uses Shapley values from game theory:
    - Treat each word as a "player" in a "game" (the prediction)
    - Each player's Shapley value = their fair contribution to the outcome
    - Positive SHAP value = word pushes prediction toward "toxic"
    - Negative SHAP value = word pushes prediction toward "not toxic"

    Args:
        texts: List of comments to explain
        predict_fn: Function that takes a list of strings → array of probabilities
        label_names: Names of the labels

    Returns:
        SHAP Explanation object
    """
    import shap

    explainer = shap.Explainer(predict_fn, shap.maskers.Text())
    shap_values = explainer(texts)

    return shap_values


def extract_attention(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int = -1,
) -> np.ndarray:
    """Extract attention weights from a transformer model.

    Attention weights show which tokens the model "looks at" when
    processing each token. High attention ≈ the model considers
    these tokens important (but this is debated in research!).

    Args:
        model: RoBERTa model
        input_ids: Tokenized input, shape (1, seq_length)
        attention_mask: Shape (1, seq_length)
        layer: Which transformer layer's attention to extract (-1 = last)

    Returns:
        Attention weights averaged across heads, shape (seq_length,)
    """
    model.eval()
    with torch.no_grad():
        outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # outputs.attentions is a tuple of (n_layers) tensors
    # Each tensor shape: (batch, n_heads, seq_len, seq_len)
    attention = outputs.attentions[layer]  # shape: (1, 12, seq_len, seq_len)

    # Average across all attention heads
    avg_attention = attention.mean(dim=1).squeeze(0)  # shape: (seq_len, seq_len)

    # Take attention FROM the [CLS] token (row 0) — what did CLS attend to?
    cls_attention = avg_attention[0].cpu().numpy()  # shape: (seq_len,)

    return cls_attention


def plot_attention(
    tokens: List[str],
    attention_weights: np.ndarray,
    title: str = "Attention Visualization",
    save_path: Optional[str] = None,
) -> None:
    """Plot attention weights as colored text highlights.

    Args:
        tokens: List of token strings
        attention_weights: Attention score per token
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Normalize to [0, 1]
    weights = attention_weights[:len(tokens)]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14)

    cmap = plt.cm.Reds
    x_pos = 0
    for i, (token, weight) in enumerate(zip(tokens, weights)):
        color = cmap(weight)
        ax.text(
            x_pos, 0.5, token + ' ',
            fontsize=11,
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
        )
        x_pos += len(token) * 0.15 + 0.5

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_lime_explanation(
    explanation,
    label_idx: int = 0,
    label_name: str = "toxic",
    save_path: Optional[str] = None,
) -> None:
    """Plot LIME feature importance as a horizontal bar chart.

    Args:
        explanation: LIME Explanation object
        label_idx: Which label to visualize
        label_name: Display name for the label
        save_path: Optional path to save
    """
    exp_list = explanation.as_list(label=label_idx)
    words = [x[0] for x in exp_list]
    weights = [x[1] for x in exp_list]

    colors = ['red' if w > 0 else 'steelblue' for w in weights]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(words)), weights, color=colors, alpha=0.8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Feature importance')
    ax.set_title(f'LIME Explanation — "{label_name}" prediction')
    ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

**Step 2: Verify**

```bash
python -c "from src.explain import lime_explain, shap_explain, extract_attention; print('OK')"
```

**Step 3: Commit**

```bash
git add src/explain.py
git commit -m "feat: add explainability module with LIME, SHAP, and attention viz"
```

---

## Task 11: Notebook 04 — Explainability Dashboard

**Files:**
- Create: `notebooks/04_explainability.ipynb`

**Notebook structure:**

1. **Why explainability matters** — real-world scenarios (content moderation, bias auditing)
2. **Load the best model** (RoBERTa) and baseline (LogReg) for comparison
3. **Concept: LIME** — explain the algorithm, show diagram
4. **LIME on 3-5 example comments** — toxic, clean, borderline
5. **Concept: SHAP** — Shapley values, explain the game theory intuition
6. **SHAP on same examples** — compare with LIME
7. **Concept: Attention** — what it shows, why it's NOT a perfect explanation
8. **Attention visualization on examples** — color-coded token highlights
9. **Compare all 3 methods** — where they agree and disagree
10. **Bias check** — do models flag identity terms (e.g., "gay", "muslim") regardless of context?
11. **Final model comparison table** — all 3 models, side by side
12. **Generate final results chart** — the image for README/resume
13. **Learning journal**
14. **Interview prep** — 5 questions (including "can you trust attention weights?")

**Step 1: Write the notebook**

**Step 2: Run notebook**

**Step 3: Commit**

```bash
git add notebooks/04_explainability.ipynb results/
git commit -m "feat: Phase 4 complete — LIME/SHAP/attention explainability"
```

---

## Task 12: Final Test Set Evaluation + README

**Files:**
- Run test set evaluation in notebook 04 (final cells)
- Create: `README.md`

**Step 1: Add final evaluation cells to notebook 04**

Run all 3 models on the held-out TEST set (first and only time we touch it).
Generate the final comparison table and save to `results/final_comparison.json`.

**Step 2: Write README.md**

```markdown
# Toxic Content Detection with Explainability

Multilabel toxicity classifier trained on the Jigsaw dataset (160K Wikipedia comments, 6 categories)...

## Results

[Final comparison table + chart]

## Architecture Progression

[Brief description of each phase with motivation]

## Key Techniques

- Stratified multilabel splitting
- Focal loss for class imbalance
- GloVe embeddings with BiLSTM
- RoBERTa fine-tuning with gradient accumulation
- LIME/SHAP/Attention explainability

## How to Run

[Setup instructions]
```

**Step 3: Final commit**

```bash
git add README.md results/ notebooks/
git commit -m "docs: add README with final results and project summary"
```

---

## Summary of Tasks

| Task | What | Key Deliverable |
|------|------|-----------------|
| 0 | Project setup | git init, .gitignore, venv, data download |
| 1 | `src/dataset.py` | Data loading + stratified splitting |
| 2 | `src/metrics.py` | Evaluation functions + comparison tables |
| 3 | Notebook 01 | EDA + TF-IDF baseline |
| 4 | `src/models.py` (BiLSTM) | BiLSTM class + GloVe loader |
| 5 | `src/training.py` | Training loop + focal loss + early stopping |
| 6 | PyTorch Dataset | Vocabulary, ToxicDataset classes |
| 7 | Notebook 02 | BiLSTM + GloVe training + evaluation |
| 8 | `src/models.py` (RoBERTa) | RoBERTa wrapper + tokenized dataset |
| 9 | Notebook 03 | RoBERTa fine-tuning + 3-model comparison |
| 10 | `src/explain.py` | LIME/SHAP/attention wrappers |
| 11 | Notebook 04 | Explainability dashboard |
| 12 | Final evaluation + README | Test set results + project documentation |
