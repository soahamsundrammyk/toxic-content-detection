"""
Evaluation metrics for multilabel toxicity classification.

Provides consistent evaluation across all model phases:
- Per-label F1, precision, recall, ROC-AUC
- Macro-averaged metrics
- Model comparison tables
- Saving/loading results for cross-phase comparison
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
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

    This is the main function you'll call after every model. It computes
    precision, recall, F1, and ROC-AUC for each of the 6 labels, plus
    the macro (overall) average.

    Args:
        y_true: Ground truth labels, shape (n_samples, n_labels)
                Each value is 0 or 1.
                Example for 2 samples, 6 labels:
                [[1, 0, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0]]

        y_pred: Binary predictions from the model, same shape as y_true.
                Also 0s and 1s — what the model THINKS the labels are.

        y_proba: Predicted probabilities, same shape. Values between 0 and 1.
                 Example: [[0.92, 0.03, 0.87, 0.01, 0.78, 0.12], ...]
                 Optional — needed for ROC-AUC but not for F1.

        label_names: Names of the labels (default: our 6 toxicity labels)

    Returns:
        Dictionary with structure:
        {
            'per_label': {
                'toxic': {'f1': 0.72, 'precision': 0.80, 'recall': 0.65, 'roc_auc': 0.95, 'support': 1534},
                'severe_toxic': {...},
                ...
            },
            'macro': {'f1': 0.58, 'precision': 0.65, 'recall': 0.52, 'roc_auc': 0.91}
        }
    """
    results = {
        'per_label': {},
        'macro': {},
    }

    # We'll collect scores for each label, then average them for macro
    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []

    for i, label in enumerate(label_names):
        # y_true[:, i] grabs column i — all samples for this one label
        # Think of it like: from a spreadsheet, grab the entire "toxic" column
        #
        # f1_score compares the true column vs predicted column
        # zero_division=0 means: if there are no positive samples, return 0
        # instead of crashing
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)

        label_metrics = {
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            # 'support' = how many positive samples exist for this label
            # int() converts numpy int to regular Python int (for JSON saving)
            'support': int(y_true[:, i].sum()),
        }

        # ROC-AUC needs probabilities, not just 0/1 predictions
        if y_proba is not None:
            auc = roc_auc_score(y_true[:, i], y_proba[:, i])
            label_metrics['roc_auc'] = round(auc, 4)
            auc_scores.append(auc)

        results['per_label'][label] = label_metrics
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)

    # Macro average = simple average across all labels
    # Each label counts equally, regardless of how many samples it has
    # np.mean([0.8, 0.7, 0.9, 0.3, 0.6, 0.5]) → averages all 6
    results['macro'] = {
        'f1': round(np.mean(f1_scores), 4),
        'precision': round(np.mean(precision_scores), 4),
        'recall': round(np.mean(recall_scores), 4),
    }

    if auc_scores:
        results['macro']['roc_auc'] = round(np.mean(auc_scores), 4)

    return results


def print_metrics(results: Dict, model_name: str = "Model") -> None:
    """Pretty-print evaluation metrics as a formatted table.

    Example output:
    ==================================================================
      TF-IDF + LogReg — Evaluation Results
    ==================================================================
              Label |     F1 |   Prec |    Rec |    AUC | Support
    ------------------------------------------------------------------
              toxic | 0.7234 | 0.8012 | 0.6589 | 0.9521 |    1534
        severe_toxic | 0.3891 | 0.5123 | 0.3145 | 0.9812 |     160
    ...
    """
    print(f"\n{'=' * 70}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 70}")

    # Build header dynamically based on whether ROC-AUC is available
    header = f"{'Label':>15s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s}"
    # Check if the first label has roc_auc
    first_label = list(results['per_label'].values())[0]
    has_auc = 'roc_auc' in first_label
    if has_auc:
        header += f" | {'AUC':>6s}"
    header += f" | {'Support':>7s}"

    print(header)
    print("-" * 70)

    # Print one row per label
    for label, metrics in results['per_label'].items():
        line = (
            f"{label:>15s} | "
            f"{metrics['f1']:6.4f} | "
            f"{metrics['precision']:6.4f} | "
            f"{metrics['recall']:6.4f}"
        )
        if has_auc:
            line += f" | {metrics['roc_auc']:6.4f}"
        line += f" | {metrics['support']:>7d}"
        print(line)

    # Print macro average row
    print("-" * 70)
    macro = results['macro']
    line = (
        f"{'MACRO AVG':>15s} | "
        f"{macro['f1']:6.4f} | "
        f"{macro['precision']:6.4f} | "
        f"{macro['recall']:6.4f}"
    )
    if 'roc_auc' in macro:
        line += f" | {macro['roc_auc']:6.4f}"
    line += f" |"
    print(line)
    print(f"{'=' * 70}\n")


def save_results(results: Dict, model_name: str, results_dir: str = 'results') -> None:
    """Save evaluation results to a JSON file for later comparison.

    Each model gets its own file: results/tfidf_logreg_results.json, etc.

    Args:
        results: Output from evaluate_predictions()
        model_name: Name like 'tfidf_logreg', 'bilstm_glove', 'roberta'
        results_dir: Directory to save to
    """
    path = Path(results_dir)
    # mkdir with exist_ok=True means: create the folder if it doesn't exist,
    # but don't crash if it already does
    path.mkdir(exist_ok=True)

    # Combine model name with the metrics
    output = {'model': model_name, **results}
    # ** unpacks a dictionary. So if results = {'per_label': ..., 'macro': ...}
    # then {'model': 'tfidf', **results} becomes
    # {'model': 'tfidf', 'per_label': ..., 'macro': ...}

    filepath = path / f"{model_name}_results.json"

    # json.dump writes a Python dictionary to a JSON file
    # indent=2 makes it human-readable (not all on one line)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {filepath}")


def load_all_results(results_dir: str = 'results') -> List[Dict]:
    """Load all saved result JSON files for comparison.

    Returns a list of result dictionaries, one per model.
    """
    path = Path(results_dir)
    results = []
    # .glob('*_results.json') finds all files matching that pattern
    # sorted() so they always come in the same order
    for filepath in sorted(path.glob('*_results.json')):
        with open(filepath) as f:
            results.append(json.load(f))
    return results


def print_comparison_table(results_list: List[Dict]) -> None:
    """Print a side-by-side comparison of all models.

    This produces the final comparison table — the one that goes on your resume.
    It shows macro F1 and AUC for each model, plus the hardest labels.

    Example output:
    ================================================================================
      MODEL COMPARISON
    ================================================================================
    Model                     | Macro F1 | Macro AUC |      threat F1 | identity_hate F1
    --------------------------------------------------------------------------------
    TF-IDF + LogReg           |   0.5821 |    0.9134 |          0.2891 |           0.3456
    BiLSTM + GloVe            |   0.6234 |    0.9456 |          0.3567 |           0.4123
    Fine-tuned RoBERTa        |   0.7012 |    0.9789 |          0.5234 |           0.5678
    """
    if not results_list:
        print("No results to compare.")
        return

    print(f"\n{'=' * 80}")
    print(f"  MODEL COMPARISON")
    print(f"{'=' * 80}")

    # Header row
    header = f"{'Model':<25s} | {'Macro F1':>8s} | {'Macro AUC':>9s}"
    # Also show the two hardest labels (rarest = hardest to predict well)
    for label in ['threat', 'identity_hate']:
        header += f" | {label + ' F1':>15s}"
    print(header)
    print("-" * 80)

    # One row per model
    for result in results_list:
        name = result['model']
        macro_f1 = result['macro']['f1']
        macro_auc = result['macro'].get('roc_auc', None)

        line = f"{name:<25s} | {macro_f1:>8.4f} | "
        if macro_auc is not None:
            line += f"{macro_auc:>9.4f}"
        else:
            line += f"{'N/A':>9s}"

        for label in ['threat', 'identity_hate']:
            label_f1 = result['per_label'][label]['f1']
            line += f" | {label_f1:>15.4f}"

        print(line)

    print(f"{'=' * 80}\n")
