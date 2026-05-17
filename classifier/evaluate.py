# classifier/evaluate.py
# ─────────────────────────────────────────────────────────────────────
# Evaluation utilities for tabular classifiers.
# Compatible with all models built by train.build_model().
# ─────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.metrics import f1_score
from loguru import logger


def evaluate_f1(model, dataset, device="cpu") -> float:
    """
    Computes macro-averaged F1 on *dataset*.

    Parameters
    ----------
    model   : Fitted sklearn-compatible classifier.
    dataset : tuple (X, y).
    device  : Unused; kept for API compatibility with neural backends.
    """
    X, y = dataset
    preds = model.predict(X)
    f1 = f1_score(y, preds, average="macro")
    logger.debug(f"Eval F1 (macro): {f1:.4f}")
    return float(f1)


def get_probabilities(model, tokenizer, X, device="cpu") -> np.ndarray:
    """
    Returns class probability matrix of shape (N, num_classes).

    All models built by build_model() expose predict_proba — SVM via Platt
    scaling (probability=True), tree ensembles natively.  This function
    raises a clear error if a custom model is passed that lacks the method.

    Parameters
    ----------
    model     : Fitted sklearn-compatible classifier.
    tokenizer : Unused; kept for API compatibility with neural backends.
    X         : Feature matrix (numpy array or DataFrame).
    device    : Unused; kept for API compatibility.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"[evaluate] {type(model).__name__} does not support predict_proba. "
            "For SVM, ensure the model was built with probability=True."
        )
    probs = model.predict_proba(X)
    return probs
