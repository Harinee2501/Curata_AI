# features/verifier.py
# -------------------------------------------------------------------
# Signal 3: Verifier Confidence
#
# Intuition: synthetic samples may have noisy or wrong labels.
# We use the classifier's own confidence in the *assigned* label
# as a proxy for label correctness.  If the model strongly agrees
# with the label, the sample is probably reliable.
#
# Note: in a real system you could use a *separate* verifier model
# (trained on a held-out set) to avoid the classifier confirming its
# own biases.  We use the same model here for simplicity.
# -------------------------------------------------------------------

import numpy as np
from loguru import logger


def compute_verifier_confidence(
    probs: np.ndarray,
    assigned_labels: list[int],
) -> np.ndarray:
    """
    Returns the classifier's probability for the assigned label of
    each synthetic sample.

    Parameters
    ----------
    probs            : np.ndarray (N, num_classes)  softmax probabilities
    assigned_labels  : list[int] of length N         synthetic label column

    Returns
    -------
    confidence : np.ndarray (N,) in [0, 1]
        Probability assigned to the label claimed by the generator.
        High = label is likely correct.

    FIX: label indices are clipped to [0, num_classes-1] before indexing
    to avoid silent IndexError on label encoding mismatches (e.g. when a
    synthetic row carries a label string that maps to an out-of-range int).
    """
    num_classes = probs.shape[1]
    indices = np.clip(np.array(assigned_labels), 0, num_classes - 1)

    if not np.array_equal(indices, np.array(assigned_labels)):
        logger.warning(
            "[verifier] Some assigned_labels were out of range for "
            f"num_classes={num_classes} — clipped to valid range. "
            "Check label encoding in _label_str_to_int."
        )

    confidence = probs[np.arange(len(probs)), indices]
    return confidence
