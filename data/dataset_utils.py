# data/dataset_utils.py  (patched)
# Change summary:
#   • load_user_csv now returns two extra values at the end of its tuple:
#       ohe         — the fitted OneHotEncoder (None for pure-numeric datasets)
#       cat_cols    — the original categorical column names that were encoded
#     These are needed by rows_dicts_to_matrix so synthetic row dicts (which
#     carry raw string values like workclass="Private") can be OHE-transformed
#     identically to real rows before the feature matrix is built.
#
#   All other logic is unchanged.

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from config import cfg
from data.row_features import (
    encoder_input_from_series,
    infer_text_partition,
)

CAT_UNIQUE_THRESHOLD = 20
CAT_RATIO_THRESHOLD = 0.50
IMBALANCE_WARN_THRESHOLD = 0.15
SMOTE_MIN_ROWS = 50


def _load_file(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext in {".csv"}:
        df = pd.read_csv(file_path)
    elif ext in {".tsv"}:
        df = pd.read_csv(file_path, sep="\t")
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif ext in {".json"}:
        df = pd.read_json(file_path, orient="records")
    else:
        logger.warning(f"Unknown extension '{ext}' — attempting to read as CSV.")
        df = pd.read_csv(file_path)
    logger.info(
        f"Loaded '{p.name}' [{ext}] — {len(df):,} rows, {len(df.columns)} columns"
    )
    return df


def _partition_columns(
    df: pd.DataFrame,
    content_cols: list[str],
    resolved_text_col: str | None,
) -> tuple[list[str], list[str]]:
    non_text = [c for c in content_cols if c != resolved_text_col]
    num_cols, cat_cols = [], []
    for col in non_text:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(col)
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            numeric_frac = coerced.notna().mean()
            if numeric_frac > 0.98:
                df[col] = coerced
                num_cols.append(col)
            else:
                cat_cols.append(col)
    return num_cols, cat_cols


def _impute(df, num_cols, cat_cols):
    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        df[num_cols] = imp_num.fit_transform(df[num_cols])
    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols].astype(str))
    return df


def _encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, OneHotEncoder | None, list[str]]:
    if not cat_cols:
        return df, None, []
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = ohe.fit_transform(df[cat_cols].astype(str))
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    df = df.drop(columns=cat_cols)
    ohe_df = pd.DataFrame(encoded, columns=ohe_names, index=df.index)
    df = pd.concat([df, ohe_df], axis=1)
    logger.info(
        f"OneHot-encoded {len(cat_cols)} categorical column(s) "
        f"→ {len(ohe_names)} binary features: {cat_cols}"
    )
    return df, ohe, ohe_names


def _scale_numeric(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def _check_and_handle_imbalance(X_train, y_train, label_names, *, use_smote=True):
    counts = np.bincount(y_train, minlength=len(label_names))
    total = len(y_train)
    minority_frac = counts.min() / max(total, 1)
    ratios = {label_names[i]: int(counts[i]) for i in range(len(label_names))}
    logger.info(f"Class distribution (train): {ratios}")
    if minority_frac < IMBALANCE_WARN_THRESHOLD:
        logger.warning(
            f"Class imbalance detected — minority class is only "
            f"{minority_frac:.1%} of training data. Distribution: {ratios}"
        )
        if use_smote and total >= SMOTE_MIN_ROWS:
            try:
                from imblearn.over_sampling import SMOTE

                k = max(1, min(5, counts.min() - 1))
                sm = SMOTE(random_state=cfg.RANDOM_SEED, k_neighbors=k)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                new_counts = np.bincount(y_train, minlength=len(label_names))
                logger.info(
                    f"SMOTE applied — new class distribution: "
                    f"{ {label_names[i]: int(new_counts[i]) for i in range(len(label_names))} }"
                )
            except ImportError:
                logger.warning("imbalanced-learn not installed — skipping SMOTE.")
            except Exception as exc:
                logger.warning(f"SMOTE failed ({exc}) — continuing without resampling.")
    else:
        logger.info(
            f"Class balance OK — minority fraction {minority_frac:.1%} "
            f"(threshold {IMBALANCE_WARN_THRESHOLD:.0%})."
        )
    return X_train, y_train


def _stratified_split(X, y, enc_inputs, *, val_ratio, test_ratio, random_seed):
    n = len(X)
    test_abs = max(1, int(n * test_ratio))
    val_abs = max(1, int(n * val_ratio))
    try:
        X_tv, X_test, y_tv, y_test, *enc_parts = _split_with_enc(
            X,
            y,
            enc_inputs,
            test_size=test_abs,
            random_state=random_seed,
            stratify=y,
        )
        X_trainval, X_val, y_trainval, y_val, *enc_parts2 = _split_with_enc(
            X_tv,
            y_tv,
            enc_parts[0] if enc_parts else None,
            test_size=val_abs,
            random_state=random_seed,
            stratify=y_tv,
        )
    except ValueError:
        logger.warning("Stratified split failed — falling back to random split.")
        X_tv, X_test, y_tv, y_test, *enc_parts = _split_with_enc(
            X,
            y,
            enc_inputs,
            test_size=test_abs,
            random_state=random_seed,
            stratify=None,
        )
        X_trainval, X_val, y_trainval, y_val, *enc_parts2 = _split_with_enc(
            X_tv,
            y_tv,
            enc_parts[0] if enc_parts else None,
            test_size=val_abs,
            random_state=random_seed,
            stratify=None,
        )
    n_train = int(len(X_trainval) * 0.4)
    X_train = X_trainval[:n_train]
    y_train = y_trainval[:n_train]
    enc_train = enc_parts2[0][:n_train] if enc_parts2 and enc_parts2[0] else None
    return X_train, y_train, X_val, y_val, X_test, y_test, enc_train


def _split_with_enc(X, y, enc_inputs, **kwargs):
    if enc_inputs is not None:
        idx = np.arange(len(X))
        idx_a, idx_b = train_test_split(idx, **kwargs)
        return (
            X[idx_a],
            X[idx_b],
            y[idx_a],
            y[idx_b],
            [enc_inputs[i] for i in idx_a],
            [enc_inputs[i] for i in idx_b],
        )
    else:
        X_a, X_b, y_a, y_b = train_test_split(X, y, **kwargs)
        return X_a, X_b, y_a, y_b, None, None


def load_user_csv(file_path: str, label_col: str, text_col: str | None = None):
    """
    Returns
    -------
    df, X_train, y_train, X_val, y_val, X_test, y_test,
    label_names, num_classes, content_cols,
    feature_mode, resolved_text_col, text_numeric_cols, train_texts_for_novelty,
    ohe,        ← NEW: fitted OneHotEncoder (None when no cat cols)
    cat_cols    ← NEW: original categorical column names that were encoded

    ``ohe`` and ``cat_cols`` must be forwarded to ``rows_dicts_to_matrix``
    (via ``pipeline.rows_to_X``) so synthetic row dicts are encoded
    identically to real rows before the feature matrix is built.
    """

    # ── 1. Load ──────────────────────────────────────────────────────
    df = _load_file(file_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {list(df.columns)}")
    if len(df) < cfg.MIN_REAL_SAMPLES:
        raise ValueError(
            f"Dataset has only {len(df)} rows; minimum is {cfg.MIN_REAL_SAMPLES}."
        )

    # ── 2. Drop missing labels ────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with missing label.")

    # ── 3. Encode labels ──────────────────────────────────────────────
    label_names = sorted(df[label_col].astype(str).unique().tolist())
    num_classes = len(label_names)
    logger.info(f"Detected {num_classes} classes: {label_names}")
    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}
    df["_label_int"] = df[label_col].astype(str).map(label_to_idx)

    content_cols = [c for c in df.columns if c not in {label_col, "_label_int"}]
    logger.info(f"Content columns ({len(content_cols)}): {content_cols}")
    if not content_cols:
        raise ValueError("No feature columns found — only the label column is present.")

    # ── 4. Partition columns ─────────────────────────────────────────
    resolved_text_col, _legacy_numeric_side = infer_text_partition(
        df, content_cols, text_col
    )
    feature_mode = "text" if resolved_text_col is not None else "tabular"
    num_cols, cat_cols = _partition_columns(df, content_cols, resolved_text_col)

    if cat_cols:
        logger.info(f"Categorical columns to OHE: {cat_cols}")

    # ── 5. Impute ─────────────────────────────────────────────────────
    df = _impute(df, num_cols, cat_cols)

    # ── 6. Encode categoricals ────────────────────────────────────────
    df, ohe, ohe_col_names = _encode_categoricals(df, cat_cols)
    effective_num_cols = num_cols + ohe_col_names

    # ── 7. Build feature matrix X ─────────────────────────────────────
    if feature_mode == "tabular":
        X = df[effective_num_cols].values.astype(np.float64)
        enc_inputs = None
        logger.info(f"Feature mode: tabular — {len(effective_num_cols)} features.")
    else:
        from features.novelty import embed

        logger.info(
            f"Feature mode: text — doc col '{resolved_text_col}', "
            f"numeric side: {effective_num_cols or '(none)'}."
        )
        enc_inputs = [
            encoder_input_from_series(df.iloc[i], resolved_text_col, effective_num_cols)
            for i in range(len(df))
        ]
        E_all = embed(enc_inputs)
        if effective_num_cols:
            M_all = df[effective_num_cols].values.astype(np.float64)
            X = np.hstack([E_all, M_all])
        else:
            X = E_all

    y = df["_label_int"].values.astype(np.int64)

    # ── 8. Stratified split ───────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, enc_train = _stratified_split(
        X,
        y,
        enc_inputs,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        random_seed=cfg.RANDOM_SEED,
    )
    enc_inputs_train = enc_train if enc_inputs is not None else None

    # ── 9. Scale ──────────────────────────────────────────────────────
    if feature_mode == "tabular":
        X_train, X_val, X_test, _scaler = _scale_numeric(X_train, X_val, X_test)
        logger.info("Applied StandardScaler to numeric features.")
    else:
        if effective_num_cols:
            d_emb = X_train.shape[1] - len(effective_num_cols)
            _scaler = StandardScaler()
            X_train[:, d_emb:] = _scaler.fit_transform(X_train[:, d_emb:])
            X_val[:, d_emb:] = _scaler.transform(X_val[:, d_emb:])
            X_test[:, d_emb:] = _scaler.transform(X_test[:, d_emb:])
            logger.info("Scaled numeric side features (text mode).")

    # ── 10. Imbalance / SMOTE ─────────────────────────────────────────
    use_smote = getattr(cfg, "USE_SMOTE", True)
    X_train, y_train = _check_and_handle_imbalance(
        X_train, y_train, label_names, use_smote=use_smote
    )

    logger.info(
        f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    text_numeric_cols = effective_num_cols if feature_mode == "text" else []
    train_texts_for_novelty = enc_inputs_train if feature_mode == "text" else None
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
        content_cols,  # original col names (for generation prompts)
        feature_mode,
        resolved_text_col,
        text_numeric_cols,
        train_texts_for_novelty,
        ohe,  # ← NEW: fitted OneHotEncoder or None
        cat_cols,  # ← NEW: original cat col names (e.g. ["workclass","sex"])
    )
