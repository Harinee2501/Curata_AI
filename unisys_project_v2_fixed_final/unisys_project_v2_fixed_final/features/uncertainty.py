# features/uncertainty.py
# -------------------------------------------------------------------
# Signal 1: Classifier Uncertainty
#
# Intuition: if the model is unsure about a sample (high entropy over
# predicted class probabilities), that sample likely lies near a
# decision boundary and is most informative for training.
#
# Formula:  H(p) = - sum_c  p_c * log(p_c + ε)
# -------------------------------------------------------------------

import numpy as np


def compute_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Computes the prediction entropy for each sample.

    Parameters
    ----------
    probs : np.ndarray of shape (N, num_classes)
        Softmax probabilities from the classifier.

    Returns
    -------
    entropy : np.ndarray of shape (N,)
        Entropy values in [0, log(num_classes)].
        Higher = more uncertain = more informative.
    """
    eps     = 1e-9                              # avoid log(0)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)

    # Normalise to [0, 1] so it's comparable with the other signals
    max_entropy = np.log(probs.shape[1])        # = log(num_classes)
    if max_entropy > 0:
        entropy = entropy / max_entropy

    return entropy
