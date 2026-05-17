# features/novelty.py
# ─────────────────────────────────────────────────────────────────────
# Sentence-embedding helpers and novelty scoring.
#
# Changes from previous version:
#   • tabular_X_to_texts / tabular_dict_to_text now guard against
#     content_cols that contain OHE-expanded column names (binary floats).
#     Those columns are included in the string representation unchanged,
#     so the embedding space is still consistent between real and synthetic
#     rows even when OHE columns are present.
#   • No other structural changes — the embedding model, L2-normalisation,
#     and cosine-distance novelty formula are unchanged.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import cfg
from data.row_features import encoder_input_from_dict

# ── module-level singleton ────────────────────────────────────────────
_embedder: SentenceTransformer | None = None

# Cap reference real rows so novelty embedding stays bounded on large CSVs.
_MAX_REAL_REF = 512


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"[novelty] Loading embedding model '{cfg.EMBEDDING_MODEL}' …")
        _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embedder


# ─────────────────────────────────────────────────────────────────────
# Core embedding
# ─────────────────────────────────────────────────────────────────────


def embed(texts: list[str]) -> np.ndarray:
    """Encode a list of strings into L2-normalised embedding vectors."""
    embedder = _get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-9)


# ─────────────────────────────────────────────────────────────────────
# Novelty computation
# ─────────────────────────────────────────────────────────────────────


def compute_novelty(
    real_embeddings: np.ndarray,
    synth_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Novelty score ∈ [0, 1] per synthetic row = 1 − max cosine similarity
    to any real training row (higher → more novel / dissimilar).
    """
    similarity_matrix = synth_embeddings @ real_embeddings.T
    max_similarity = similarity_matrix.max(axis=1)
    return np.clip(1.0 - max_similarity, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# Tabular ↔ text conversion helpers
# ─────────────────────────────────────────────────────────────────────


def tabular_X_to_texts(X: np.ndarray, content_cols: list[str]) -> list[str]:
    """
    Convert numeric training rows to short strings for the sentence encoder.

    Works correctly when *content_cols* includes OHE-expanded binary columns
    (they appear as "col_name=value: 0.0" / "… 1.0" in the string).
    """
    lines: list[str] = []
    for i in range(len(X)):
        parts = [f"{c}: {X[i, j]:.6g}" for j, c in enumerate(content_cols)]
        lines.append(" | ".join(parts))
    return lines


def tabular_dict_to_text(row: dict, content_cols: list[str]) -> str:
    """
    Same layout as ``tabular_X_to_texts`` for one synthetic row dict.

    Missing columns default to 0 so real and synthetic rows are encoded
    in the same embedding space.
    """
    return " | ".join(f"{c}: {row.get(c, 0)}" for c in content_cols)


# ─────────────────────────────────────────────────────────────────────
# Main novelty API
# ─────────────────────────────────────────────────────────────────────


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
    Per-synthetic-row novelty in [0, 1]: cosine distance from the nearest
    real training embedding (1 − max cosine similarity).

    Uses the same sentence encoder as the rest of the project. The real
    reference set is sub-sampled to ``_MAX_REAL_REF`` rows on large CSVs.

    In ``feature_mode="text"``, ``train_texts`` (encoder strings aligned
    with ``X_train`` rows) and ``text_col`` / ``text_numeric_cols`` are
    used so synthetic rows are encoded in the same space as real ones.

    Fallback: if ``feature_mode="text"`` but ``train_texts`` or ``text_col``
    is None, both real and synthetic rows are encoded using the tabular
    ``col: val`` format so their embeddings remain in a consistent space.
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

    use_text_mode = (
        feature_mode == "text" and train_texts is not None and text_col is not None
    )

    if use_text_mode:
        side = text_numeric_cols or []
        real_texts = [train_texts[i] for i in idx]
        synth_texts = [encoder_input_from_dict(r, text_col, side) for r in synth_rows]
    else:
        if feature_mode == "text" and (train_texts is None or text_col is None):
            logger.warning(
                "[novelty] feature_mode='text' but train_texts or text_col is None — "
                "falling back to tabular encoding for novelty. Both real and synthetic "
                "rows will be encoded as 'col: val | …' strings."
            )
        real_texts = tabular_X_to_texts(ref_X, content_cols)
        synth_texts = [tabular_dict_to_text(r, content_cols) for r in synth_rows]

    real_emb = embed(real_texts)
    synth_emb = embed(synth_texts)
    return np.asarray(compute_novelty(real_emb, synth_emb), dtype=np.float64)
