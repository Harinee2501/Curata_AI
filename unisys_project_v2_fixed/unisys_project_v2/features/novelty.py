import numpy as np
from sentence_transformers import SentenceTransformer
from config import cfg   # ✅ FIX: import cfg instead of config module

# Load the embedding model once at module level
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        print(f"[novelty] Loading embedding model '{cfg.EMBEDDING_MODEL}' ...")
        _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """Encodes a list of texts into L2-normalised embedding vectors."""
    embedder   = _get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # L2-normalise so cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-9)


def compute_novelty(
    real_embeddings: np.ndarray,
    synth_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Computes novelty score based on cosine distance from nearest real sample.
    """
    similarity_matrix = synth_embeddings @ real_embeddings.T
    max_similarity = similarity_matrix.max(axis=1)
    novelty = np.clip(1.0 - max_similarity, 0.0, 1.0)
    return novelty