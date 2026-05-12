# data/dataset_utils.py
# ─────────────────────────────────────────────────────────────────────
# Tabular dataset loader (NO TEXT, NO TOKENIZER)
# ─────────────────────────────────────────────────────────────────────

import random
import pandas as pd
import numpy as np
from loguru import logger
from config import cfg


def load_user_csv(csv_path: str, label_col: str):

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

    # 👉 NUMERIC FEATURES
    X = df[content_cols].values.astype(float)
    y = df["_label_int"].values

    # shuffle
    combined = list(zip(X, y))
    random.seed(cfg.RANDOM_SEED)
    random.shuffle(combined)

    X, y = zip(*combined)
    X = np.array(X)
    y = np.array(y)

    n = len(X)
    n_val = int(n * cfg.VAL_RATIO)
    n_test = int(n * cfg.TEST_RATIO)
    n_train = int((n - n_val - n_test) * 0.4)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    logger.info(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    df = df.drop(columns=["_label_int"])

    return (
        df,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_names, num_classes, content_cols,
    )