"""
Dataset loading and splitting for Jigsaw Toxic Comment Classification.

This module handles:
1. Loading the CSV data
2. Basic text cleaning
3. Stratified train/val/test splitting
4. Vocabulary building (word → index mapping)
5. PyTorch Dataset classes for deep learning
"""

import re
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# The 6 toxicity labels in the Jigsaw dataset.
# We define this ONCE here so every other file can import it.
# This avoids typos — if you mistype 'severre_toxic' somewhere,
# you'd get a silent bug. Using this list everywhere prevents that.
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load the Jigsaw training CSV and do basic validation.

    Args:
        data_dir: Path to the directory containing train.csv

    Returns:
        DataFrame with columns: id, comment_text, + 6 label columns
    """
    # Path is from Python's pathlib — a nicer way to handle file paths
    # Path('data') / 'train.csv' creates 'data/train.csv'
    # It works on Mac, Windows, Linux without worrying about / vs \
    path = Path(data_dir) / 'train.csv'

    if not path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {path}. "
            "Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"
        )

    # pd.read_csv reads a CSV file into a DataFrame (a table)
    df = pd.read_csv(path)

    # Some comments might be missing (NaN). Fill them with empty string
    # so we don't crash later when trying to process text.
    # .fillna('') means: wherever there's a missing value, replace with ''
    df['comment_text'] = df['comment_text'].fillna('')

    return df


def clean_text(text: str) -> str:
    """Minimal text cleaning — lowercase and normalize whitespace.

    We keep it minimal ON PURPOSE. Why?
    - ALL CAPS often means anger/shouting — that's a useful signal!
    - Punctuation like "!!!" can indicate toxicity
    - But we lowercase because "IDIOT" and "idiot" should be the same word
    - And we normalize whitespace because "hello    world" and "hello world" are the same

    For the TF-IDF baseline we'll use this. For BERT later, we won't —
    BERT has its own tokenizer that handles everything.

    Args:
        text: Raw comment text

    Returns:
        Cleaned text
    """
    # .lower() converts "HELLO World" → "hello world"
    text = text.lower()

    # re.sub(pattern, replacement, string) does regex substitution
    # \s+ matches one or more whitespace characters (spaces, tabs, newlines)
    # We replace them all with a single space, then .strip() removes
    # leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_splits(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with stratified sampling.

    WHY 3 SPLITS?
    - Train (80%): the model learns from this data
    - Val (10%): we check performance while developing (to tune settings)
    - Test (10%): we check performance ONCE at the very end (the honest score)

    If we only had train + test, we'd be tempted to keep tweaking the model
    until the test score looks good — but then the test score is no longer
    honest. The val set is our "practice exam" that we can look at freely.

    WHY STRATIFIED?
    Imagine you randomly split and by bad luck, ALL the 'threat' comments
    (only ~0.3% of data!) end up in the training set and NONE in test.
    Your test evaluation for 'threat' would be meaningless.

    Stratification ensures each split has roughly the SAME proportion
    of each label combination.

    Args:
        df: Full dataset DataFrame
        test_size: Fraction for test set (default 0.1 = 10%)
        val_size: Fraction for validation set (default 0.1 = 10%)
        random_state: Seed for reproducibility (same number = same split every time)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # --- Step 1: Create a stratification key ---
    # We join all 6 labels into a single string like "100100"
    # This way, a comment that is [toxic=1, severe_toxic=0, obscene=0,
    # threat=1, insult=0, identity_hate=0] becomes "100100"
    #
    # .astype(str) converts 1 → "1" and 0 → "0"
    # .agg(''.join, axis=1) joins them across columns for each row
    stratify_key = df[LABEL_COLS].astype(str).agg(''.join, axis=1)

    # --- Step 2: Handle rare combinations ---
    # Some label combos might only appear once (e.g., "000011").
    # train_test_split with stratify will CRASH if any group has < 2 samples.
    # So we replace those rare combos with a generic "rare" label.
    #
    # .value_counts() counts how many times each combo appears
    combo_counts = stratify_key.value_counts()
    # Keep only combos that appear fewer than 2 times
    rare_combos = combo_counts[combo_counts < 2].index
    # Replace them with "rare"
    stratify_key = stratify_key.replace(rare_combos, 'rare')

    # --- Step 3: First split → 80% train, 20% temp ---
    temp_size = test_size + val_size  # 0.1 + 0.1 = 0.2

    # train_test_split is from scikit-learn
    # It shuffles the data and splits it into two parts
    # stratify=stratify_key ensures proportional representation
    train_df, temp_df, strat_train, strat_temp = train_test_split(
        df, stratify_key,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_key,
    )

    # --- Step 4: Second split → split the 20% into 10% val + 10% test ---
    val_fraction = val_size / temp_size  # 0.1 / 0.2 = 0.5

    # Rebuild stratify key for the temp subset
    strat_temp_key = strat_temp.reset_index(drop=True)
    temp_df = temp_df.reset_index(drop=True)

    # Handle rare combos again (some combos might be rare in this smaller set)
    temp_combo_counts = strat_temp_key.value_counts()
    rare_temp = temp_combo_counts[temp_combo_counts < 2].index
    strat_temp_key = strat_temp_key.replace(rare_temp, 'rare')

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_fraction),  # 0.5 → half goes to test
        random_state=random_state,
        stratify=strat_temp_key,
    )

    # --- Step 5: Clean up indices ---
    # After splitting, the index numbers are jumbled (e.g., [4502, 123, 99821...])
    # reset_index makes them clean: [0, 1, 2, 3, ...]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a summary of the data splits showing sizes and label distributions.

    This is a sanity check — we want to see that:
    1. The sizes are roughly 80/10/10
    2. Each label's percentage is similar across all 3 splits
       (that means stratification worked!)
    """
    total = len(train_df) + len(val_df) + len(test_df)

    print(f"{'Split':<8} {'Size':>7} {'% of total':>10}")
    print("-" * 28)
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pct = len(split_df) / total * 100
        print(f"{name:<8} {len(split_df):>7,} {pct:>9.1f}%")

    print(f"\nLabel distribution (% positive) per split:")
    print(f"{'Label':<15} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 40)
    for label in LABEL_COLS:
        # .mean() on a column of 0s and 1s gives the percentage of 1s
        # e.g., [0, 0, 1, 0, 1].mean() = 0.4 = 40%
        train_pct = train_df[label].mean() * 100
        val_pct = val_df[label].mean() * 100
        test_pct = test_df[label].mean() * 100
        print(f"{label:<15} {train_pct:>6.2f}% {val_pct:>6.2f}% {test_pct:>6.2f}%")


# ============================================================================
# PYTORCH DATA PIPELINE (for Phase 2+)
# ============================================================================


class Vocabulary:
    """Maps words to integer indices and back.

    Why do we need this?
    Neural networks can't process strings like "idiot" — they need numbers.
    The Vocabulary creates a mapping:
        "idiot"  → 42
        "stupid" → 891
        "hello"  → 7
        (unknown word) → 1  (the <UNK> token)
        (padding)      → 0  (the <PAD> token)

    Special tokens:
        <PAD> = 0  — used to fill short sequences to equal length.
                     All comments must be the same length in a batch,
                     so short ones get padded with zeros.
        <UNK> = 1  — used for words not in our vocabulary.
                     If the model sees a word during validation that
                     it never saw during training, it maps to <UNK>.
    """

    def __init__(self, max_size: int = 50000):
        """
        Args:
            max_size: Maximum vocabulary size.
                      Only keep the most common words.
                      Same idea as max_features in TF-IDF.
        """
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.max_size = max_size

    def build(self, texts: pd.Series) -> 'Vocabulary':
        """Build vocabulary from a pandas Series of text.

        IMPORTANT: Only call this on TRAINING data!
        If we included val/test words, that would be data leakage —
        the model would "know" about words it shouldn't have seen yet.

        Args:
            texts: pandas Series of comment strings

        Returns:
            self (so you can chain: vocab = Vocabulary().build(texts))
        """
        # Counter counts how many times each word appears
        # e.g., Counter({'the': 490031, 'to': 294069, 'idiot': 1523, ...})
        counter = Counter()
        for text in texts:
            words = clean_text(text).split()
            counter.update(words)

        # Keep only the most common words (minus 2 for PAD and UNK)
        # .most_common(N) returns the N most frequent words
        for word, count in counter.most_common(self.max_size - 2):
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        print(f"Vocabulary built: {len(self.word_to_idx):,} words "
              f"(from {len(counter):,} unique words in training data)")
        return self

    def encode(self, text: str, max_length: int = 200) -> list:
        """Convert text to a list of integer indices.

        Steps:
            1. Clean the text (lowercase, normalize whitespace)
            2. Split into words
            3. Map each word to its index (or <UNK> if unknown)
            4. Truncate if too long
            5. Pad with zeros if too short

        Args:
            text: Input comment text
            max_length: All sequences will be exactly this long.
                        Longer → truncated. Shorter → padded with 0s.
                        200 words covers ~95% of comments without wasting memory.

        Returns:
            List of integers, always exactly max_length long.

        Example:
            vocab.encode("you are an idiot", max_length=6)
            → [42, 8, 15, 891, 0, 0]
               ↑   ↑   ↑   ↑   ↑  ↑
              you  are  an idiot PAD PAD
        """
        words = clean_text(text).split()

        # Convert words to indices
        # .get(word, 1) means: look up the word, if not found return 1 (<UNK>)
        indices = [self.word_to_idx.get(w, 1) for w in words]

        # Truncate if longer than max_length
        indices = indices[:max_length]

        # Pad with zeros if shorter than max_length
        padding_needed = max_length - len(indices)
        indices = indices + [0] * padding_needed

        return indices

    def __len__(self) -> int:
        """Number of words in the vocabulary."""
        return len(self.word_to_idx)


class ToxicDataset(Dataset):
    """PyTorch Dataset for the Jigsaw toxic comment data.

    A Dataset is like a smart list. PyTorch's DataLoader calls:
        dataset[0]  → returns first comment (encoded) + its labels
        dataset[42] → returns comment #42 (encoded) + its labels
        len(dataset) → how many comments total

    The DataLoader then groups these into batches:
        batch = [dataset[0], dataset[1], ..., dataset[31]]  (batch_size=32)
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_length: int = 200):
        """
        Args:
            df: DataFrame with 'comment_text' and the 6 label columns
            vocab: Vocabulary object for encoding text → indices
            max_length: Maximum sequence length (pad/truncate to this)
        """
        self.texts = df['comment_text'].tolist()
        # .astype(np.float32) because PyTorch expects float, not int, for labels
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        """How many samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get one sample by index.

        This is called by DataLoader when it builds a batch.

        Args:
            idx: Index of the sample (0 to len-1)

        Returns:
            Dictionary with:
              'input_ids': tensor of word indices, shape (max_length,)
              'labels': tensor of 6 label values, shape (6,)
        """
        text = self.texts[idx]
        indices = self.vocab.encode(text, self.max_length)

        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
