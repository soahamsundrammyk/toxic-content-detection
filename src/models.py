"""
Neural network model definitions for toxicity classification.

Phase 2: BiLSTM with GloVe embeddings
Phase 3: RoBERTa fine-tuning wrapper
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for multilabel text classification.

    Architecture:
        text indices → embedding → BiLSTM → dropout → linear → (logits)

    How data flows through this model:
        1. Input: [42, 891, 7, 2203, 0, 0, ...]  (word indices, padded)
        2. Embedding: each index → 100-dim GloVe vector
        3. BiLSTM: reads sequence forward AND backward
        4. Take final hidden states from both directions, concatenate
        5. Dropout: randomly zero out 30% of values (training only)
        6. Linear: map to 6 toxicity label scores
        7. Output: [2.1, -0.3, 1.8, -1.5, 0.9, -0.2] (raw logits)

    We output LOGITS (raw scores), not probabilities.
    The loss function applies sigmoid internally for numerical stability.
    For predictions, apply torch.sigmoid() separately.
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
            vocab_size: Number of unique words in our vocabulary.
                        Determines the size of the embedding lookup table.

            embedding_dim: Size of each word vector.
                           100 for GloVe 6B 100d.
                           Think of it as: each word is described by 100 numbers.

            hidden_dim: Number of features in the LSTM hidden state.
                        128 means each direction produces a 128-dim summary.
                        Bigger = more capacity to learn, but slower and more overfitting risk.

            num_labels: Number of output labels (6 for our Jigsaw dataset).

            num_layers: Number of stacked LSTM layers.
                        2 means we stack two LSTMs on top of each other.
                        Layer 1 reads the word embeddings.
                        Layer 2 reads Layer 1's output — learns higher-level patterns.

            dropout: Probability of zeroing out values during training.
                     0.3 = randomly turn off 30% of neurons each forward pass.
                     This forces the model to not rely on any single feature,
                     reducing overfitting.

            pretrained_embeddings: Optional numpy array of GloVe vectors,
                                   shape (vocab_size, embedding_dim).
                                   If provided, we use these instead of random vectors.
        """
        # super().__init__() calls the parent class (nn.Module) constructor.
        # This is required boilerplate for all PyTorch models.
        super().__init__()

        # --- LAYER 1: Embedding ---
        # nn.Embedding is a lookup table: index → vector
        # It holds a matrix of shape (vocab_size, embedding_dim)
        # When you pass index 42, it returns row 42 of the matrix.
        #
        # padding_idx=0 means: index 0 (our <PAD> token) always maps to
        # a vector of all zeros. Padding should have no meaning.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # If we have pre-trained GloVe vectors, load them into the embedding table
        if pretrained_embeddings is not None:
            # .weight.data is the actual matrix inside nn.Embedding
            # We copy the GloVe vectors into it
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Freeze: do NOT update GloVe vectors during training.
            # Why? GloVe was trained on 6 BILLION words — our 127K comments
            # are not enough to improve these vectors, and we might make them worse.
            self.embedding.weight.requires_grad = False

        # --- LAYER 2: BiLSTM ---
        # nn.LSTM handles all the gate math (forget, input, output) internally.
        # We just tell it the dimensions and it does the rest.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,   # each input is a 100-dim word vector
            hidden_size=hidden_dim,     # each direction produces 128-dim output
            num_layers=num_layers,      # stack 2 LSTM layers
            batch_first=True,           # input shape: (batch, seq_len, features)
                                        # (not (seq_len, batch, features))
            bidirectional=True,         # read forward AND backward
            dropout=dropout if num_layers > 1 else 0,
            # ^ dropout BETWEEN stacked layers (not at the end)
            #   Only applies when num_layers > 1
        )

        # --- LAYER 3: Dropout ---
        # nn.Dropout randomly zeros out elements with probability p during training.
        # During evaluation (model.eval()), dropout is automatically disabled.
        self.dropout = nn.Dropout(dropout)

        # --- LAYER 4: Linear (fully connected) layer ---
        # Maps from LSTM output to 6 label predictions.
        #
        # Input size = hidden_dim * 2 because BiLSTM concatenates forward + backward.
        # Each direction gives 128 dims → concatenated = 256 dims.
        # Output size = 6 (one score per toxicity label).
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: text indices → toxicity logits.

        This method defines what happens when data passes through the model.
        PyTorch calls this automatically when you do: output = model(input)

        Args:
            x: Token indices, shape (batch_size, seq_length)
               Example: tensor([[42, 891, 7, 0, 0],
                                [12, 55, 893, 221, 0]])
               Each number is a word index. 0 = padding.

        Returns:
            Logits (raw scores), shape (batch_size, num_labels)
            Example: tensor([[2.1, -0.3, 1.8, -1.5, 0.9, -0.2],
                             [0.5, -1.2, 0.3, -0.8, 0.1, -0.5]])
        """
        # Step 1: Embedding lookup
        # x shape: (batch_size, seq_length)  e.g., (32, 200)
        # Each number gets replaced by its 100-dim vector
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)  e.g., (32, 200, 100)

        # Step 2: BiLSTM
        # lstm_out contains the output at EVERY time step
        # hidden contains the FINAL hidden state
        # cell contains the FINAL cell state (the "notebook")
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_length, hidden_dim * 2)  e.g., (32, 200, 256)
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)  e.g., (4, 32, 128)
        #   num_layers * 2 because: 2 layers × 2 directions = 4
        #   The 4 entries are: [layer1_forward, layer1_backward, layer2_forward, layer2_backward]

        # Step 3: Extract final hidden states from the LAST layer
        # hidden[-2] = last layer, forward direction
        # hidden[-1] = last layer, backward direction
        forward_hidden = hidden[-2]    # shape: (batch_size, hidden_dim) e.g., (32, 128)
        backward_hidden = hidden[-1]   # shape: (batch_size, hidden_dim) e.g., (32, 128)

        # Concatenate forward and backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        # combined shape: (batch_size, hidden_dim * 2)  e.g., (32, 256)

        # Step 4: Dropout (only active during training)
        dropped = self.dropout(combined)

        # Step 5: Linear layer → 6 label scores
        logits = self.fc(dropped)
        # logits shape: (batch_size, num_labels)  e.g., (32, 6)

        return logits


def load_glove_embeddings(
    glove_path: str,
    word_to_idx: dict,
    embedding_dim: int = 100,
) -> np.ndarray:
    """Load pre-trained GloVe vectors and align them with our vocabulary.

    Our vocabulary has its own word→index mapping (built from our dataset).
    GloVe has vectors for 400K words. We need to match them up:
      - For words in BOTH our vocab AND GloVe: use the GloVe vector
      - For words in our vocab but NOT in GloVe: use random vectors
      - For words in GloVe but NOT our vocab: ignore them

    Args:
        glove_path: Path to glove.6B.100d.txt
        word_to_idx: Our vocabulary's word → index mapping
                     e.g., {'<PAD>': 0, '<UNK>': 1, 'the': 2, 'idiot': 42, ...}
        embedding_dim: Dimension of GloVe vectors (100 for 6B.100d)

    Returns:
        Numpy array of shape (vocab_size, embedding_dim)
        Row i = the embedding vector for word with index i
    """
    vocab_size = len(word_to_idx)

    # Initialize with small random values (for words not found in GloVe)
    # Why random and not zeros? Because if all unknown words had the same
    # vector [0,0,0,...], the model couldn't distinguish between them.
    embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    # Padding token (index 0) should be all zeros — padding has no meaning
    embeddings[0] = np.zeros(embedding_dim)

    # Read the GloVe file line by line
    # Each line: "word 0.123 -0.456 0.789 ..." (word followed by 100 numbers)
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]

            # Only keep vectors for words in OUR vocabulary
            if word in word_to_idx:
                idx = word_to_idx[word]
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[idx] = vector
                found += 1

    coverage = found / vocab_size * 100
    print(f"GloVe: found vectors for {found:,}/{vocab_size:,} words ({coverage:.1f}% coverage)")

    return embeddings.astype(np.float32)


# ============================================================================
# PHASE 3: RoBERTa Fine-tuning
# ============================================================================


class RoBERTaClassifier(nn.Module):
    """Fine-tuned RoBERTa for multilabel toxicity classification.

    Architecture:
        tokenized text → RoBERTa encoder → [CLS] token output → dropout → linear → logits

    The [CLS] token (first token) is designed by RoBERTa's training to be
    a summary of the whole input. We use this as our feature for classification.

    This is TRANSFER LEARNING: we load RoBERTa's pre-trained weights
    (trained on 160GB of text) and fine-tune them on our 127K comments.
    Unlike GloVe (frozen), we DO update RoBERTa's weights during training.
    """

    def __init__(
        self,
        num_labels: int = 6,
        model_name: str = 'roberta-base',
        dropout: float = 0.1,
    ):
        """
        Args:
            num_labels: Number of output labels (6 for Jigsaw)
            model_name: Which pre-trained model to use.
                        'roberta-base' = 125M parameters, 768 hidden dim
                        'roberta-large' = 355M parameters (too big for our M3)
            dropout: Dropout probability for the classification head.
                     0.1 is standard for transformer fine-tuning
                     (lower than LSTM's 0.3 because the model is already regularized
                     by pre-training).
        """
        super().__init__()

        # Import here to avoid forcing transformers install at module load
        from transformers import RobertaModel

        # Load pre-trained RoBERTa
        # This downloads ~500MB the first time (cached for future use)
        self.roberta = RobertaModel.from_pretrained(model_name)

        # RoBERTa's hidden dimension (768 for roberta-base)
        hidden_size = self.roberta.config.hidden_size

        # Dropout before the classifier
        self.dropout = nn.Dropout(dropout)

        # Classification head: a single linear layer
        # Input: 768 dims ([CLS] token vector)
        # Output: 6 dims (one score per label)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: tokenized text → toxicity logits.

        Args:
            input_ids: Tokenized input from RoBERTa tokenizer.
                       Shape: (batch_size, seq_length)
                       Each value is a token ID in RoBERTa's vocabulary.

            attention_mask: 1 for real tokens, 0 for padding.
                            Shape: (batch_size, seq_length)
                            Tells RoBERTa which positions to ignore (padding).

        Returns:
            Logits of shape (batch_size, num_labels)
        """
        # Step 1: Run RoBERTa encoder
        # outputs is a special object containing:
        #   - last_hidden_state: shape (batch_size, seq_length, 768)
        #     → one 768-dim vector per token
        #   - pooler_output: shape (batch_size, 768)
        #     → a pooled representation (not what we use)
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Step 2: Extract the [CLS] token's output (first token, index 0)
        # Shape: (batch_size, 768)
        # This vector summarizes the whole input.
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Step 3: Apply dropout
        dropped = self.dropout(cls_output)

        # Step 4: Linear classification head → logits
        # Shape: (batch_size, num_labels) = (batch_size, 6)
        logits = self.classifier(dropped)

        return logits
