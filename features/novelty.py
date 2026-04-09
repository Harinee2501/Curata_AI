# features/novelty.py
# -------------------------------------------------------------------
# Signal 2: Semantic Novelty
#
# Intuition: a synthetic sample that is semantically far from all
# real training samples introduces new coverage for the model —
# it fills gaps in the representation space.
#
# Method: encode all texts with a sentence transformer, then for each
# synthetic sample compute the cosine distance to its nearest real
# neighbour.  High distance = novel = potentially valuable.
# -------------------------------------------------------------------

import numpy as np
from sentence_transformers import SentenceTransformer
import config

# Load the embedding model once at module level so it isn't reloaded
# on every function call.
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        print(f"[novelty] Loading embedding model '{config.EMBEDDING_MODEL}' ...")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """Encodes a list of texts into L2-normalised embedding vectors."""
    embedder   = _get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-normalise so cosine similarity = dot product (faster nearest-neighbour)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-9)


def compute_novelty(
    real_embeddings: np.ndarray,
    synth_embeddings: np.ndarray,
) -> np.ndarray:
    """
    For each synthetic sample, finds the cosine distance to the
    nearest real sample.

    Parameters
    ----------
    real_embeddings  : np.ndarray (M, dim)  — embeddings of real data
    synth_embeddings : np.ndarray (N, dim)  — embeddings of synthetic candidates

    Returns
    -------
    novelty_scores : np.ndarray (N,) in [0, 1]
        0 = identical to a real sample, 1 = maximally novel.
    """
    # Dot product of normalised vectors = cosine similarity
    similarity_matrix = synth_embeddings @ real_embeddings.T  # (N, M)

    # Nearest-neighbour similarity = max across real samples
    max_similarity = similarity_matrix.max(axis=1)            # (N,)

    # Convert similarity to distance: 0 = same, 1 = opposite
    # Clamp to [0, 1] since floating-point noise can exceed bounds
    novelty = np.clip(1.0 - max_similarity, 0.0, 1.0)
    return novelty
