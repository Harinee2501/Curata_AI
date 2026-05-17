# data/row_features.py  (patched)
# Change summary:
#   • rows_dicts_to_matrix now accepts two new keyword arguments:
#       ohe      — fitted OneHotEncoder returned by load_user_csv (or None)
#       cat_cols — original categorical column names (e.g. ["workclass", "sex"])
#     When supplied, categorical string values in each row dict are OHE-
#     transformed before the numeric feature matrix is assembled.  This
#     fixes the 'workclass' KeyError / NaN-coercion bug that occurred
#     because synthetic row dicts carry raw string values ("Private",
#     "Male", …) that pd.to_numeric silently converts to NaN.
#
#   All other logic (tabular path, text path, infer_text_partition, etc.)
#   is unchanged.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def is_numeric_like_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    return bool(coerced.notna().mean() > 0.98)


def is_high_cardinality_categorical(
    s: pd.Series, *, n_unique_threshold: int = 20
) -> bool:
    if is_numeric_like_series(s):
        return False
    n_unique = s.nunique(dropna=True)
    return n_unique <= n_unique_threshold


def infer_text_partition(
    df: pd.DataFrame,
    content_cols: list[str],
    text_col: str | None = None,
) -> tuple[str | None, list[str]]:
    if text_col is not None:
        if text_col not in content_cols:
            raise ValueError(
                f"text_col '{text_col}' is not among content columns: {content_cols}."
            )
        numeric_rest = [
            c for c in content_cols if c != text_col and is_numeric_like_series(df[c])
        ]
        return text_col, numeric_rest

    non_numeric = [c for c in content_cols if not is_numeric_like_series(df[c])]

    if len(non_numeric) == 0:
        return None, list(content_cols)

    if len(non_numeric) == 1:
        tcol = non_numeric[0]
        numeric_rest = [
            c for c in content_cols if c != tcol and is_numeric_like_series(df[c])
        ]
        return tcol, numeric_rest

    best_col: str | None = None
    best_score: float = -1.0
    for col in non_numeric:
        s = df[col].dropna().astype(str)
        avg_tokens = s.str.split().str.len().mean()
        if avg_tokens < 3:
            continue
        uniq_ratio = s.nunique() / max(len(s), 1)
        score = avg_tokens * uniq_ratio
        if score > best_score:
            best_score = score
            best_col = col

    if best_col is not None and best_score > 2.0:
        numeric_rest = [
            c for c in content_cols if c != best_col and is_numeric_like_series(df[c])
        ]
        logger = _lazy_logger()
        logger.info(
            f"Auto-detected text column '{best_col}' (score={best_score:.2f}) "
            f"from non-numeric candidates: {non_numeric}."
        )
        return best_col, numeric_rest

    logger = _lazy_logger()
    logger.info(
        f"{len(non_numeric)} non-numeric columns {non_numeric} look categorical — "
        f"pure tabular/categorical mode."
    )
    return None, list(content_cols)


def _lazy_logger():
    try:
        from loguru import logger as _l

        return _l
    except ImportError:
        import logging

        return logging.getLogger(__name__)


def encoder_input_from_series(
    row: pd.Series, text_col: str, numeric_cols: list[str]
) -> str:
    t = str(row[text_col])
    if not numeric_cols:
        return t
    rest = " | ".join(f"{c}: {row[c]}" for c in numeric_cols)
    return f"{t} :: {rest}"


def encoder_input_from_dict(row: dict, text_col: str, numeric_cols: list[str]) -> str:
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
    # ── NEW: categorical encoding artefacts from load_user_csv ───────
    ohe: OneHotEncoder | None = None,
    cat_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Map a list of row dicts to design matrix X (n, d).

    Parameters
    ----------
    ohe : fitted OneHotEncoder (returned by load_user_csv), or None.
    cat_cols : the original categorical column names that were OHE-encoded
               during training (e.g. ["workclass", "education", "sex"]).

    When ``ohe`` and ``cat_cols`` are provided (tabular mode), categorical
    string values in each dict are transformed and appended to the numeric
    columns — exactly replicating what load_user_csv does for real rows.
    Without this, values like "Private" survive as NaN after pd.to_numeric,
    silently zeroing out the categorical signal for every synthetic row.
    """
    from features.novelty import embed

    if feature_mode == "tabular":
        df_tmp = pd.DataFrame(rows)

        # ── Numeric columns ──────────────────────────────────────────
        # content_cols contains the ORIGINAL column names (pre-OHE).
        # We only coerce the purely-numeric subset here; categoricals are
        # handled separately below.
        _cat_set = set(cat_cols or [])
        numeric_content_cols = [c for c in content_cols if c not in _cat_set]

        for col in numeric_content_cols:
            if col not in df_tmp.columns:
                df_tmp[col] = 0.0

        num_part = (
            (
                df_tmp[numeric_content_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .values.astype(np.float64)
            )
            if numeric_content_cols
            else np.empty((len(rows), 0), dtype=np.float64)
        )

        # ── Categorical columns (OHE) ────────────────────────────────
        if ohe is not None and cat_cols:
            for col in cat_cols:
                if col not in df_tmp.columns:
                    # Synthetic row may omit a categorical col entirely —
                    # fill with the empty string so OHE maps to the
                    # "unknown" all-zeros row (handle_unknown="ignore").
                    df_tmp[col] = ""

            cat_part = ohe.transform(df_tmp[cat_cols].astype(str)).astype(np.float64)
            return np.hstack([num_part, cat_part]) if num_part.shape[1] else cat_part

        return num_part

    # ── Text mode ────────────────────────────────────────────────────
    assert text_col is not None, "text_col must be provided in text feature mode."
    side = text_numeric_cols or []
    texts = [encoder_input_from_dict(r, text_col, side) for r in rows]
    E = embed(texts)
    if not side:
        return E

    df_tmp = pd.DataFrame(rows)
    for col in side:
        if col not in df_tmp.columns:
            df_tmp[col] = 0.0
    M = (
        df_tmp[side]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float64)
    )
    return np.hstack([E, M])
