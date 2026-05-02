# classifier/evaluate.py
# -------------------------------------------------------------------
# Evaluation utilities for tabular models (Logistic Regression)
# -------------------------------------------------------------------

import numpy as np
from sklearn.metrics import f1_score


def evaluate_f1(model, dataset, device="cpu") -> float:
    """
    dataset = (X, y)
    """
    X, y = dataset

    preds = model.predict(X)
    f1 = f1_score(y, preds, average="macro")

    return float(f1)


def get_probabilities(model, tokenizer, X, device="cpu") -> np.ndarray:
    """
    Returns (N, num_classes) probabilities
    tokenizer is unused (kept for compatibility)
    """
    probs = model.predict_proba(X)
    return probs