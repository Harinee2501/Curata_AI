# data/dataset_utils.py
# ─────────────────────────────────────────────────────────────────────
# Labeled CSV loader:
#   • Tabular — numeric features → matrix X as floats (original behaviour)
#   • Text    — one document column (+ optional numeric side features)
#               → sentence-embeddings (and optional numeric concat) for X
# ─────────────────────────────────────────────────────────────────────

import random
import numpy as np
import pandas as pd
from loguru import logger

from config import cfg
from data.row_features import (
    encoder_input_from_series,
    infer_text_partition,
)


def load_user_csv(csv_path: str, label_col: str, text_col: str | None = None):
    """
    Load a labeled CSV and build train/val/test splits.

    Parameters
    ----------
    csv_path
        Path to CSV file.
    label_col
        Target / label column name.
    text_col
        Optional. Name of the free-text document column. If omitted, auto-detect:
        exactly one non-numeric-like content column is treated as text.

    Returns
    -------
    df, X_train, y_train, X_val, y_val, X_test, y_test,
    label_names, num_classes, content_cols,
    feature_mode, resolved_text_col, text_numeric_cols, train_texts_for_novelty

    ``text_numeric_cols`` — side numeric feature names when ``feature_mode=='text'``;
    empty list in tabular mode.

    ``train_texts_for_novelty`` — encoder-input strings aligned with ``X_train``
    (for novelty vs synthetic). ``None`` in tabular mode.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV: {csv_path} — {len(df)} rows, {len(df.columns)} columns")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    if len(df) < cfg.MIN_REAL_SAMPLES:
        raise ValueError(f"Minimum {cfg.MIN_REAL_SAMPLES} rows required")

    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    label_names = sorted(df[label_col].astype(str).unique().tolist())
    num_classes = len(label_names)

    logger.info(f"Detected {num_classes} classes: {label_names}")

    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}
    df["_label_int"] = df[label_col].astype(str).map(label_to_idx)

    content_cols = [c for c in df.columns if c not in [label_col, "_label_int"]]
    logger.info(f"Content columns: {content_cols}")

    if not content_cols:
        raise ValueError("No feature columns found (only label column present).")

    resolved_text_col, numeric_side = infer_text_partition(df, content_cols, text_col)
    feature_mode = "text" if resolved_text_col is not None else "tabular"

    if feature_mode == "tabular":
        X = df[content_cols].values.astype(np.float64)
        enc_inputs = None
        logger.info("Feature mode: tabular (numeric matrix).")
    else:
        from features.novelty import embed

        logger.info(
            f"Feature mode: text embeddings — document column '{resolved_text_col}', "
            f"numeric side columns: {numeric_side or '(none)'}"
        )
        enc_inputs = [
            encoder_input_from_series(df.iloc[i], resolved_text_col, numeric_side)
            for i in range(len(df))
        ]
        E_all = embed(enc_inputs)
        if numeric_side:
            M_all = df[numeric_side].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            M_all = M_all.values.astype(np.float64)
            X = np.hstack([E_all, M_all])
        else:
            X = E_all

    y = df["_label_int"].values.astype(np.int64)

    if feature_mode == "text":
        paired = list(zip(X, y, enc_inputs))
    else:
        paired = list(zip(X, y))

    random.seed(cfg.RANDOM_SEED)
    random.shuffle(paired)

    if feature_mode == "text":
        X_rows, y_arr, enc_shuf = zip(*paired)
        X = np.stack(X_rows, axis=0)
        y = np.array(y_arr, dtype=np.int64)
        enc_inputs_shuffled = list(enc_shuf)
    else:
        X_rows, y_arr = zip(*paired)
        X = np.stack(X_rows, axis=0)
        y = np.array(y_arr, dtype=np.int64)
        enc_inputs_shuffled = None

    n = len(X)
    n_val = int(n * cfg.VAL_RATIO)
    n_test = int(n * cfg.TEST_RATIO)
    n_train = int((n - n_val - n_test) * 0.4)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    if feature_mode == "text":
        train_texts_for_novelty = enc_inputs_shuffled[:n_train]
        text_numeric_cols = list(numeric_side)
    else:
        train_texts_for_novelty = None
        text_numeric_cols = []

    logger.info(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    df = df.drop(columns=["_label_int"])

    return (
        df,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        label_names,
        num_classes,
        content_cols,
        feature_mode,
        resolved_text_col,
        text_numeric_cols,
        train_texts_for_novelty,
    )
