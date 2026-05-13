# data/row_features.py
# ─────────────────────────────────────────────────────────────────────
# Helpers for tabular vs text (sentence-embedding) feature pipelines.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd


def is_numeric_like_series(s: pd.Series) -> bool:
    """True if values are numeric dtype or parse as numbers for almost all rows."""
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    return bool(coerced.notna().mean() > 0.98)


def infer_text_partition(
    df: pd.DataFrame,
    content_cols: list[str],
    text_col: str | None = None,
) -> tuple[str | None, list[str]]:
    """
    Returns (text_col_or_none, numeric_content_cols).

    - Tabular: (None, content_cols) when every column is numeric-like.
    - Text: (doc_col, numeric_rest) when there is a single text-like column
      or when ``text_col`` names the document column (others must be numeric-like).
    """
    if text_col is not None:
        if text_col not in content_cols:
            raise ValueError(
                f"text_col '{text_col}' is not among content columns {content_cols}."
            )
        numeric_rest = [c for c in content_cols if c != text_col]
        for c in numeric_rest:
            if not is_numeric_like_series(df[c]):
                raise ValueError(
                    f"When using text_col='{text_col}', column '{c}' must be numeric-like."
                )
        return text_col, numeric_rest

    non_numeric = [c for c in content_cols if not is_numeric_like_series(df[c])]
    if len(non_numeric) == 0:
        return None, list(content_cols)
    if len(non_numeric) == 1:
        tcol = non_numeric[0]
        return tcol, [c for c in content_cols if c != tcol]
    raise ValueError(
        "Multiple non-numeric content columns found: "
        f"{non_numeric}. Specify which is the document column via text_col=..."
    )


def encoder_input_from_series(
    row: pd.Series,
    text_col: str,
    numeric_cols: list[str],
) -> str:
    """Single training row → string fed to the sentence encoder (must match dict path)."""
    t = str(row[text_col])
    if not numeric_cols:
        return t
    rest = " | ".join(f"{c}: {row[c]}" for c in numeric_cols)
    return f"{t} :: {rest}"


def encoder_input_from_dict(
    row: dict,
    text_col: str,
    numeric_cols: list[str],
) -> str:
    """Synthetic / CSV row dict → same encoder layout as ``encoder_input_from_series``."""
    t = str(row.get(text_col, ""))
    if not numeric_cols:
        return t
    rest = " | ".join(f"{c}: {row.get(c, '')}" for c in numeric_cols)
    return f"{t} :: {rest}"


def rows_dicts_to_matrix(
    rows: list[dict],
    *,
    feature_mode: str,
    content_cols: list[str],
    text_col: str | None,
    text_numeric_cols: list[str] | None,
) -> np.ndarray:
    """Maps list of row dicts to design matrix X (n, d)."""
    from features.novelty import embed

    if feature_mode == "tabular":
        return np.array([[float(r[c]) for c in content_cols] for r in rows], dtype=np.float64)

    assert text_col is not None
    side = text_numeric_cols or []
    texts = [encoder_input_from_dict(r, text_col, side) for r in rows]
    E = embed(texts)
    if not side:
        return E
    M = np.array([[float(r[c]) for c in side] for r in rows], dtype=np.float64)
    return np.hstack([E, M])
