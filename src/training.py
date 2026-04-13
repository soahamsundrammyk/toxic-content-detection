"""
Training loop, loss functions, and utilities for PyTorch models.

Includes:
- Focal Loss (handles class imbalance better than standard BCE)
- Training loop with early stopping
- Evaluation function
- Device selection (MPS for M3 Mac)
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Get the best available device for computation.

    On your M3 Mac:
      - MPS (Metal Performance Shaders) = Apple's GPU acceleration
      - It is like NVIDIA's CUDA but for Apple Silicon
      - Matrix multiplications run on the GPU → much faster than CPU

    Returns:
        torch.device — either 'mps' or 'cpu'
    """
    if torch.backends.mps.is_available():
        print("Using MPS (Apple M3 GPU)")
        return torch.device('mps')
    print("Using CPU (MPS not available)")
    return torch.device('cpu')


class FocalLoss(nn.Module):
    """Focal Loss for multilabel classification with class imbalance.

    THE PROBLEM:
        Standard BCE (Binary Cross Entropy) treats all examples equally.
        But in our data, 90% of comments are clean. The model sees way more
        "easy" clean examples than "hard" toxic ones. It spends most of its
        learning effort on things it already gets right.

    THE SOLUTION:
        Focal Loss adds a factor that:
        - DOWN-weights easy examples (model already confident and correct)
        - UP-weights hard examples (model uncertain or wrong)

    THE MATH:
        Standard BCE:  loss = -[y·log(p) + (1-y)·log(1-p)]

        Focal Loss:    loss = -α · (1-p_t)^γ · [y·log(p) + (1-y)·log(1-p)]

        Where:
          p = predicted probability
          y = actual label (0 or 1)
          p_t = probability of the CORRECT class
                (p if y=1, or 1-p if y=0)
          γ (gamma) = focusing parameter
          α (alpha) = class weight

    HOW (1-p_t)^γ WORKS:
        If model predicts correctly with high confidence:
          p_t = 0.95 → (1-0.95)^2 = 0.0025 → loss is TINY
          "You already know this, stop wasting time on it"

        If model predicts incorrectly:
          p_t = 0.1 → (1-0.1)^2 = 0.81 → loss is LARGE
          "You got this wrong! Pay attention!"

        γ=0 → standard BCE (no focusing)
        γ=2 → strong focusing (common default, what we use)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Args:
            alpha: Overall weighting factor. 1.0 = no extra weighting.
            gamma: Focusing parameter.
                   0 = identical to standard BCE
                   2 = strong focus on hard examples (recommended)
                   5 = very aggressive focusing (can be unstable)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output BEFORE sigmoid.
                    Shape: (batch_size, num_labels) e.g., (32, 6)
                    We use logits (not probabilities) because
                    binary_cross_entropy_with_logits is numerically stable.
                    It applies sigmoid internally in a way that avoids log(0).

            targets: Ground truth labels, 0 or 1.
                     Shape: (batch_size, num_labels) e.g., (32, 6)

        Returns:
            Scalar loss value (single number)
        """
        # Step 1: Compute standard BCE loss (per element, not averaged yet)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # bce_loss shape: (batch_size, num_labels) — one loss per label per sample

        # Step 2: Convert logits to probabilities for the focal weight
        probs = torch.sigmoid(logits)

        # Step 3: Compute p_t (probability of the correct class)
        # If target=1: p_t = p (we want high p)
        # If target=0: p_t = 1-p (we want low p, so 1-p should be high)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Step 4: Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Step 5: Apply focal weight to BCE loss
        loss = self.alpha * focal_weight * bce_loss

        # Step 6: Average across all samples and labels
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch (one full pass through training data).

    What happens in one epoch:
        1. Split training data into batches (e.g., 32 comments each)
        2. For each batch:
           a. Forward pass: data → model → predictions
           b. Compute loss: how wrong are the predictions?
           c. Backward pass: compute gradients (direction to adjust weights)
           d. Optimizer step: actually adjust the weights
        3. Return the average loss across all batches

    Args:
        model: The neural network (BiLSTM)
        dataloader: Serves batches of (input, labels)
        loss_fn: How to compute error (FocalLoss)
        optimizer: How to adjust weights (Adam)
        device: Where to run computation (MPS/CPU)

    Returns:
        Average training loss for this epoch
    """
    # model.train() enables training-specific behavior:
    #   - Dropout: randomly zeros out values (active)
    #   - BatchNorm: uses batch statistics (if we had it)
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Move data to the GPU (MPS) — computation happens on the device
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # --- Forward pass ---
        # Feed data through the model to get predictions
        logits = model(inputs)

        # Compute loss (how wrong are we?)
        loss = loss_fn(logits, labels)

        # --- Backward pass ---
        # optimizer.zero_grad() clears gradients from the previous batch.
        # Without this, gradients would ACCUMULATE across batches,
        # and the weight updates would be wrong.
        optimizer.zero_grad()

        # loss.backward() computes the gradient of the loss with respect
        # to every weight in the model. This is called "backpropagation."
        # After this, every weight has a .grad attribute saying
        # "if you increase this weight, the loss changes by this much."
        loss.backward()

        # Gradient clipping: if any gradient is extremely large,
        # cap it at max_norm=1.0. This prevents "exploding gradients"
        # which can happen in RNNs/LSTMs and cause NaN values.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Update weights ---
        # optimizer.step() adjusts every weight based on its gradient.
        # For Adam: new_weight = old_weight - adaptive_lr * gradient
        optimizer.step()

        total_loss += loss.item()  # .item() converts tensor → Python number
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()  # Decorator: disable gradient computation (saves memory + speed)
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluate model on a dataset (validation or test).

    Similar to training but:
      - No gradient computation (saves memory)
      - No weight updates
      - Dropout is disabled (model.eval())
      - We collect predictions for metric computation

    Args:
        model: The neural network
        dataloader: Validation/test data batches
        loss_fn: Loss function (same as training, for comparable loss values)
        device: MPS/CPU

    Returns:
        Dictionary with:
          'loss': average loss
          'predictions': binary 0/1 predictions (threshold 0.5)
          'probabilities': raw probabilities (for ROC-AUC)
          'labels': actual ground truth labels
    """
    # model.eval() disables training-specific behavior:
    #   - Dropout: all values pass through (no random zeroing)
    #   - This ensures consistent evaluation
    model.eval()

    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(inputs)
        loss = loss_fn(logits, labels)

        # Convert logits → probabilities using sigmoid
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        n_batches += 1

        # Move results back to CPU and convert to numpy for sklearn metrics
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Concatenate all batches into single arrays
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return {
        'loss': total_loss / n_batches,
        'probabilities': all_probs,
        'predictions': (all_probs >= 0.5).astype(int),  # threshold at 0.5
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

    EARLY STOPPING:
        We track the validation loss after each epoch.
        If it does not improve for `patience` consecutive epochs,
        we stop training and restore the best model.

        Why? Without this, the model would keep training and start
        MEMORIZING the training data (overfitting). The val loss
        would start going UP even though train loss goes DOWN.

        Typical pattern:
          Epoch 1: train_loss=0.5, val_loss=0.48  ← learning
          Epoch 2: train_loss=0.3, val_loss=0.35  ← learning
          Epoch 3: train_loss=0.2, val_loss=0.33  ← learning (best!)
          Epoch 4: train_loss=0.1, val_loss=0.34  ← val got worse (patience 1)
          Epoch 5: train_loss=0.08, val_loss=0.36 ← worse again (patience 2)
          Epoch 6: train_loss=0.05, val_loss=0.38 ← worse again (patience 3)
          → STOP. Restore model from epoch 3 (the best one).

    Args:
        model: The neural network
        train_loader: Training data batches
        val_loader: Validation data batches
        loss_fn: Loss function (FocalLoss)
        optimizer: Optimizer (Adam)
        device: MPS or CPU
        n_epochs: Maximum number of epochs to train
        patience: Stop after this many epochs without val improvement
        save_path: Optional file path to save the best model checkpoint

    Returns:
        Dictionary with training history:
          {'train_loss': [...], 'val_loss': [...]}
    """
    best_val_loss = float('inf')  # infinity — anything will be better
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': []}
    best_state = None

    print(f"Training for up to {n_epochs} epochs (patience={patience})...")
    print(f"{'Epoch':>6s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'Time':>6s} | {'Status':>15s}")
    print("-" * 60)

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # Evaluate on validation set
        val_results = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_results['loss']

        elapsed = time.time() - start_time

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save a copy of the model weights (on CPU to save GPU memory)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = "* new best *"
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            status = f"no improve ({epochs_without_improvement}/{patience})"

        print(
            f"{epoch+1:>6d} | "
            f"{train_loss:>10.4f} | "
            f"{val_loss:>10.4f} | "
            f"{elapsed:>5.1f}s | "
            f"{status:>15s}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs.")
            break

    # Restore the best model
    if best_state is not None:
        model.load_state_dict(best_state)
        # Move model back to device (we saved on CPU)
        model.to(device)
        print(f"Restored best model (val_loss: {best_val_loss:.4f})")

    return history


# ============================================================================
# PHASE 3: RoBERTa training functions
# ============================================================================
#
# RoBERTa needs attention_mask in addition to input_ids.
# We also use gradient accumulation to simulate larger batch sizes
# without running out of memory.


def train_one_epoch_roberta(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    scheduler: Optional[object] = None,
) -> float:
    """Train RoBERTa for one epoch with gradient accumulation.

    GRADIENT ACCUMULATION:
        Instead of updating weights after every batch of 8,
        we accumulate gradients over multiple batches before updating.

        accumulation_steps=2 means:
          batch 1: compute gradients, DO NOT update weights
          batch 2: compute gradients, ADD to saved gradients, UPDATE weights
          (effective batch size = 8 × 2 = 16)

        This lets us get the benefit of larger batches without the memory cost.

    Args:
        model: RoBERTa classifier
        dataloader: Training batches (with input_ids, attention_mask, labels)
        loss_fn: Loss function (FocalLoss)
        optimizer: AdamW optimizer
        device: MPS or CPU
        accumulation_steps: How many batches to accumulate before updating
        scheduler: Optional learning rate scheduler (step after each update)

    Returns:
        Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Clear gradients once at the start
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move all three tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        # Normalize loss by accumulation steps
        # This keeps the gradient magnitude similar regardless of accumulation_steps
        loss = loss / accumulation_steps

        # Backward pass: compute gradients (but do NOT update yet)
        loss.backward()

        total_loss += loss.item() * accumulation_steps  # un-normalize for logging
        n_batches += 1

        # Update weights every `accumulation_steps` batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Clip gradients to prevent instability in transformers
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Apply accumulated gradients
            optimizer.step()

            # Update learning rate schedule (if using one)
            if scheduler is not None:
                scheduler.step()

            # Clear for next accumulation cycle
            optimizer.zero_grad()

    return total_loss / n_batches


@torch.no_grad()
def evaluate_roberta(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluate RoBERTa on a dataset.

    Similar to evaluate() but handles attention_mask.
    """
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


def train_roberta(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 3,
    patience: int = 2,
    accumulation_steps: int = 2,
    scheduler: Optional[object] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """Full training loop for RoBERTa with early stopping and gradient accumulation.

    Transformer fine-tuning needs FEWER epochs than LSTM training:
      - The model is already well-trained (pre-training)
      - We just need to adapt it to our task
      - 2-4 epochs usually enough
      - More epochs risk overfitting and destroying pre-trained knowledge

    Args:
        model: RoBERTa classifier
        train_loader: Training batches
        val_loader: Validation batches
        loss_fn: FocalLoss or BCE
        optimizer: AdamW (with low learning rate like 2e-5)
        device: MPS or CPU
        n_epochs: Max epochs (transformers need fewer — 3 is typical)
        patience: Early stopping patience (2 is typical)
        accumulation_steps: Gradient accumulation steps
        scheduler: Optional learning rate scheduler
        save_path: Optional path to save best model

    Returns:
        Training history dict
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': []}
    best_state = None

    print(f"Training RoBERTa for up to {n_epochs} epochs (patience={patience})...")
    print(f"Gradient accumulation: {accumulation_steps} steps "
          f"(effective batch size = {train_loader.batch_size * accumulation_steps})")
    print(f"{'Epoch':>6s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'Time':>6s} | {'Status':>15s}")
    print("-" * 65)

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train_one_epoch_roberta(
            model, train_loader, loss_fn, optimizer, device,
            accumulation_steps=accumulation_steps,
            scheduler=scheduler,
        )

        val_results = evaluate_roberta(model, val_loader, loss_fn, device)
        val_loss = val_results['loss']

        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = "* new best *"
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            status = f"no improve ({epochs_without_improvement}/{patience})"

        print(
            f"{epoch+1:>6d} | "
            f"{train_loss:>10.4f} | "
            f"{val_loss:>10.4f} | "
            f"{elapsed:>5.1f}s | "
            f"{status:>15s}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Restored best model (val_loss: {best_val_loss:.4f})")

    return history
