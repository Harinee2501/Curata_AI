import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import cfg
from data.row_features import encoder_input_from_dict

# Load the embedding model once at module level
_embedder = None

# Cap reference real rows so novelty embedding stays bounded on large CSVs.
_MAX_REAL_REF = 512


def _get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"[novelty] Loading embedding model '{cfg.EMBEDDING_MODEL}' ...")
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


def tabular_X_to_texts(X: np.ndarray, content_cols: list[str]) -> list[str]:
    """Turn numeric training rows into short strings for the sentence encoder."""
    lines: list[str] = []
    for i in range(len(X)):
        parts = [f"{c}: {X[i, j]}" for j, c in enumerate(content_cols)]
        lines.append(" | ".join(parts))
    return lines


def tabular_dict_to_text(row: dict, content_cols: list[str]) -> str:
    """Same layout as `tabular_X_to_texts` for one synthetic CSV row dict."""
    return " | ".join(f"{c}: {row.get(c, '')}" for c in content_cols)


def novelty_scores_for_synthetic_vs_train(
    X_train: np.ndarray,
    content_cols: list[str],
    synth_rows: list[dict],
    *,
    feature_mode: str = "tabular",
    text_col: str | None = None,
    text_numeric_cols: list[str] | None = None,
    train_texts: list[str] | None = None,
) -> np.ndarray:
    """
    Per-synthetic-row novelty in [0, 1]: distance from nearest *real* training
    embedding (1 − max cosine similarity). Uses the same encoder as the rest
    of the project; real reference is subsampled when |train| is large.

    In ``feature_mode="text"``, pass ``train_texts`` (encoder strings aligned with
    ``X_train`` rows) and ``text_col`` / ``text_numeric_cols`` so synthetic rows
    are encoded the same way as training data.
    """
    n = len(synth_rows)
    if n == 0 or X_train is None or len(X_train) == 0 or not content_cols:
        return np.full(n, 0.5, dtype=np.float64)

    ref_X = np.asarray(X_train, dtype=np.float64)
    idx = np.arange(len(ref_X))
    if len(ref_X) > _MAX_REAL_REF:
        rng = np.random.default_rng(int(cfg.RANDOM_SEED))
        idx = rng.choice(len(ref_X), size=_MAX_REAL_REF, replace=False)
        ref_X = ref_X[idx]

    if feature_mode == "text" and train_texts is not None and text_col is not None:
        side = text_numeric_cols or []
        real_texts = [train_texts[i] for i in idx]
        synth_texts = [
            encoder_input_from_dict(r, text_col, side) for r in synth_rows
        ]
    else:
        real_texts = tabular_X_to_texts(ref_X, content_cols)
        synth_texts = [tabular_dict_to_text(r, content_cols) for r in synth_rows]

    real_emb = embed(real_texts)
    synth_emb = embed(synth_texts)
    return np.asarray(compute_novelty(real_emb, synth_emb), dtype=np.float64)