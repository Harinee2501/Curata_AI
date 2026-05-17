# pipeline.py — RL-integrated curation pipeline
#
# TWO MODES:
#   1. RL mode    — agent starts from ZERO rows and decides everything:
#                   when to generate, how much, when to filter,
#                   when to retrain, and when to stop.
#                   No pre-generation. No naive baseline run upfront.
#
#   2. Bandit mode — original Thompson bandit loop (fallback).
#                   Pre-generates a pool (target_size × pool_multiplier),
#                   then filters it with the Thompson bandit.
#
# Switch RL vs bandit by placing/removing the PPO zip named in config.yaml
# (default: best_model.zip) in the project root.
#
# Changes from previous version:
#   - run_pipeline / _run_rl_mode / _run_bandit_mode all accept an optional
#     `classifier_model` kwarg that temporarily overrides cfg.CLASSIFIER_MODEL
#     for the duration of that pipeline run. This lets callers (e.g. api.py)
#     choose the classifier per-request without mutating shared config state
#     for concurrent jobs.
#   - run_baseline accepts `classifier_model` and forwards it to build_model.
#   - _bandit_filter: accept arm now uses ThompsonBandit.compute_accept_reward()
#     (tiered v3: ver>0.55 → 1.0, ver>0.40 → 0.5, else 0.0).
#   - _bandit_filter: reject arm now uses ThompsonBandit.compute_reject_reward()
#     (ver<0.30 → 0.8, ver<0.40 → 0.3, else 0.0).
#   - run_pipeline (bandit mode): pool is now generated at
#     target_size × cfg.BANDIT_POOL_MULTIPLIER so the bandit has candidates
#     to filter rather than exactly the number of rows you want to keep.
#   - Naive augmentation baseline deferred — only computed if the bandit
#     produces at least one curated row, avoiding a wasted train+eval when
#     the pool is poor.

import sys, os, json, random
import functools
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))

from config import cfg
from data.dataset_utils import load_user_csv
from data.generate_synthetic import generate_synthetic_pool
from data.row_features import rows_dicts_to_matrix
from classifier.train import build_model, train
from classifier.evaluate import evaluate_f1, get_probabilities
from features.uncertainty import compute_uncertainty
from features.verifier import compute_verifier_confidence
from bandit.thompson import ThompsonBandit


def _scaled_delta_f1_reward(prev_f1: float, new_f1: float) -> float:
    """Continuous bandit reward from validation F1 change (scaled + clipped)."""
    d = float(new_f1) - float(prev_f1)
    return float(
        np.clip(
            d * cfg.THOMPSON_REWARD_SCALE,
            -cfg.THOMPSON_REWARD_CLIP,
            cfg.THOMPSON_REWARD_CLIP,
        )
    )


def setup_logging(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def _label_str_to_int(label_str, label_names):
    return {l: i for i, l in enumerate(label_names)}.get(str(label_str), 0)


def _resolve_classifier(classifier_model: str | None) -> str:
    """
    Return the active classifier key:
      - caller-supplied value (validated) takes priority
      - falls back to cfg.CLASSIFIER_MODEL (set from config.yaml)
    Logs a single INFO line so runs are traceable.
    """
    from config import VALID_CLASSIFIERS

    if classifier_model:
        clf = classifier_model.lower().strip()
        if clf not in VALID_CLASSIFIERS:
            raise ValueError(
                f"[pipeline] Unknown classifier_model '{clf}'. "
                f"Choose one of: {', '.join(sorted(VALID_CLASSIFIERS))}"
            )
    else:
        clf = cfg.CLASSIFIER_MODEL
    logger.info(f"Classifier: {clf}")
    return clf


def run_baseline(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    extra_X=None,
    extra_y=None,
    label="Baseline",
    classifier_model: str | None = None,
):
    """
    Train a fresh model and report val + test F1.

    Parameters
    ----------
    classifier_model : override the classifier for this run only.
                       Falls back to cfg.CLASSIFIER_MODEL when None.
    """
    clf = _resolve_classifier(classifier_model)

    if extra_X is not None:
        X_train = np.vstack([X_train, extra_X])
        y_train = np.concatenate([y_train, extra_y])

    # Temporarily patch cfg so build_model picks up the right key.
    _orig = cfg.CLASSIFIER_MODEL
    cfg.CLASSIFIER_MODEL = clf
    try:
        model = build_model(cfg.NUM_CLASSES)
        logger.info(f"── {label} ──")
        train(model, (X_train, y_train))
        val_f1 = evaluate_f1(model, (X_val, y_val))
        test_f1 = evaluate_f1(model, (X_test, y_test))
        logger.info(f"Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    finally:
        cfg.CLASSIFIER_MODEL = _orig

    return {"val_f1": val_f1, "test_f1": test_f1}


def save_outputs(curated_rows, original_df, results, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    curated_df = pd.DataFrame(curated_rows)
    curated_df.to_csv(out / "curated.csv", index=False)
    pd.concat([original_df, curated_df]).to_csv(out / "augmented.csv", index=False)
    with open(out / "report.json", "w") as f:
        json.dump(results, f, indent=2)


def _bandit_filter(
    rows,
    model,
    content_cols,
    label_col,
    label_names,
    get_probs_fn,
    X_train=None,
    *,
    feature_mode: str = "tabular",
    text_col: str | None = None,
    text_numeric_cols: list[str] | None = None,
    train_texts: list[str] | None = None,
    bandit: ThompsonBandit | None = None,
):
    """
    Thompson accept/reject with context [uncertainty, novelty, verifier].
    Novelty uses sentence embeddings vs real training rows (see features.novelty).

    ``bandit`` — reuse the same ``ThompsonBandit`` across FILTER calls (RL episode).
    If ``None``, a fresh bandit is created (standalone / backward-compatible use).

    Reward scheme (v3 thresholds — matches ThompsonBandit methods):
        Accept arm  — bandit.compute_accept_reward(ver):
            ver > 0.55  → 1.0   (good row, weak-model regime)
            ver > 0.40  → 0.5   (borderline — partial credit)
            ver ≤ 0.40  → 0.0   (likely noise — no credit)

        Reject arm  — bandit.compute_reject_reward(ver):
            ver < 0.30  → 0.8   (correctly rejected bad row)
            ver < 0.40  → 0.3   (minor credit)
            ver ≥ 0.40  → 0.0   (decent row rejected — no credit)
    """
    if bandit is None:
        bandit = ThompsonBandit(context_boost=cfg.THOMPSON_CONTEXT_BOOST)
    accepted, verifier_scores = [], []
    novelties_accepted: list[float] = []

    try:
        from features.novelty import novelty_scores_for_synthetic_vs_train

        if X_train is not None and len(rows) > 0:
            novelty_vec = novelty_scores_for_synthetic_vs_train(
                X_train,
                content_cols,
                rows,
                feature_mode=feature_mode,
                text_col=text_col,
                text_numeric_cols=text_numeric_cols or [],
                train_texts=train_texts,
            )
        else:
            novelty_vec = np.full(len(rows), 0.5, dtype=np.float64)
    except Exception as e:
        logger.warning(f"Novelty disabled (using 0.5): {e}")
        novelty_vec = np.full(len(rows), 0.5, dtype=np.float64)

    X_mat = rows_dicts_to_matrix(
        rows,
        feature_mode=feature_mode,
        content_cols=content_cols,
        text_col=text_col,
        text_numeric_cols=text_numeric_cols,
    )

    for i, row in enumerate(rows):
        x = X_mat[i : i + 1]
        y = _label_str_to_int(row[label_col], label_names)
        probs = get_probs_fn(model, None, x)
        unc = compute_uncertainty(probs)[0]
        ver = compute_verifier_confidence(probs, [y])[0]
        novelty = float(novelty_vec[i])
        action = (
            1
            if i < cfg.BANDIT_WARM_START
            else bandit.select(np.array([unc, novelty, ver], dtype=np.float64))
        )
        if action == 1:
            accepted.append(row)
            verifier_scores.append(float(ver))
            novelties_accepted.append(novelty)

        # Warm-start accepts explore without shifting the Thompson posterior.
        # Non-warm: use tiered rewards for both arms so the bandit learns to
        # discriminate rather than almost-always accepting.
        if i >= cfg.BANDIT_WARM_START:
            if action == 1:
                r = bandit.compute_accept_reward(ver)
                bandit.update(1, r)
            else:
                r = bandit.compute_reject_reward(ver)
                bandit.update(0, r)

    acc_rate = len(accepted) / max(len(rows), 1)
    vc_mean = float(np.mean(verifier_scores)) if verifier_scores else 0.5
    summary = bandit.summary()
    summary["novelty_mean_pool"] = (
        float(np.mean(novelty_vec)) if len(novelty_vec) else None
    )
    summary["novelty_mean_accepted"] = (
        float(np.mean(novelties_accepted)) if novelties_accepted else None
    )
    return accepted, acc_rate, vc_mean, summary


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────


# pipeline.py — patch: unpack ohe + cat_cols from load_user_csv and
# thread them into rows_to_X so synthetic row dicts are OHE-transformed
# before the feature matrix is assembled.
#
# ONLY the sections that changed are shown here.
# Apply these as a diff to the existing pipeline.py.
# ─────────────────────────────────────────────────────────────────────
#
# ── CHANGE 1 ─────────────────────────────────────────────────────────
# In run_pipeline, unpack the two new values returned by load_user_csv:
#
# BEFORE:
#     ) = load_user_csv(csv_path, label_col, text_col=text_col)
#
# AFTER:
#     ohe,
#     cat_cols,
# ) = load_user_csv(csv_path, label_col, text_col=text_col)
#
# ── CHANGE 2 ─────────────────────────────────────────────────────────
# In run_pipeline, pass ohe + cat_cols into rows_to_X via functools.partial
# or a closure so every call to rows_dicts_to_matrix receives them.
#
# BEFORE:
#     def rows_to_X(rows):
#         return rows_dicts_to_matrix(
#             rows,
#             feature_mode=feature_mode,
#             content_cols=content_cols,
#             text_col=resolved_text_col,
#             text_numeric_cols=text_numeric_cols,
#         )
#
# AFTER:
#     def rows_to_X(rows):
#         return rows_dicts_to_matrix(
#             rows,
#             feature_mode=feature_mode,
#             content_cols=content_cols,
#             text_col=resolved_text_col,
#             text_numeric_cols=text_numeric_cols,
#             ohe=ohe,          # ← NEW
#             cat_cols=cat_cols, # ← NEW
#         )
#
# That's it — no other changes to pipeline.py are required.
# _run_rl_mode and _run_bandit_mode both receive rows_to_X as a closure
# and call it wherever they need X matrices, so the fix propagates
# automatically to both modes and to _bandit_filter (via bandit_filter_fn).
# ─────────────────────────────────────────────────────────────────────

# ── FULL run_pipeline function (drop-in replacement) ─────────────────


def run_pipeline(
    csv_path,
    label_col,
    augmentation_mode: str | None = None,
    target_size: int | None = None,
    output_dir=None,
    text_col=None,
    classifier_model: str | None = None,
):
    from config import VALID_MODES

    out_dir = output_dir or cfg.OUTPUT_DIR
    setup_logging(out_dir)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    # ── Resolve augmentation mode ─────────────────────────────────────
    if augmentation_mode is not None:
        mode = augmentation_mode.lower().strip()
        if mode not in VALID_MODES:
            raise ValueError(
                f"[pipeline] Unknown augmentation_mode '{mode}'. "
                f"Choose one of: {', '.join(sorted(VALID_MODES))}"
            )
    else:
        mode = cfg.DEFAULT_AUGMENTATION_MODE

    mode_cfg = cfg.get_mode_config(mode)
    logger.info(
        f"Augmentation mode: {mode} "
        f"(pool_multiplier={mode_cfg['pool_multiplier']}, "
        f"rl_max_steps={mode_cfg['rl_max_steps']}, "
        f"patience={mode_cfg['patience']}, "
        f"curation_threshold={mode_cfg['curation_threshold']})"
    )

    _orig_clf = cfg.CLASSIFIER_MODEL
    _orig_rl_max_steps = cfg.RL_MAX_STEPS
    _orig_patience = cfg.EARLY_STOP_PATIENCE
    _orig_curation_threshold = getattr(cfg, "CURATION_THRESHOLD", None)

    clf = _resolve_classifier(classifier_model)
    cfg.CLASSIFIER_MODEL = clf
    cfg.RL_MAX_STEPS = mode_cfg["rl_max_steps"]
    cfg.EARLY_STOP_PATIENCE = mode_cfg["patience"]
    cfg.CURATION_THRESHOLD = mode_cfg["curation_threshold"]

    try:
        logger.info("Loading dataset...")
        (
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
            ohe,  # ← NEW: fitted OneHotEncoder (None for pure-numeric data)
            cat_cols,  # ← NEW: original categorical column names
        ) = load_user_csv(csv_path, label_col, text_col=text_col)
        cfg.NUM_CLASSES = num_classes

        # Derive target_size from mode when not supplied explicitly.
        if augmentation_mode is not None or target_size is None:
            target_size = cfg.pool_size_for_mode(mode, len(X_train))
            logger.info(
                f"target_size derived from mode '{mode}': "
                f"{len(X_train)} real rows × {mode_cfg['pool_multiplier']} = {target_size}"
            )

        # ── rows_to_X closure — includes ohe + cat_cols ───────────────
        # CRITICAL FIX: pass ohe and cat_cols so categorical string values
        # in synthetic row dicts (e.g. workclass="Private") are OHE-
        # transformed before the numeric matrix is assembled, exactly as
        # load_user_csv does for real rows.
        def rows_to_X(rows):
            return rows_dicts_to_matrix(
                rows,
                feature_mode=feature_mode,
                content_cols=content_cols,
                text_col=resolved_text_col,
                text_numeric_cols=text_numeric_cols,
                ohe=ohe,  # ← NEW
                cat_cols=cat_cols,  # ← NEW
            )

        rows_to_y = lambda rows: np.array(
            [_label_str_to_int(r[label_col], label_names) for r in rows]
        )

        baseline = run_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            label="Baseline — real only",
        )

        policy_path = Path(cfg.RL_POLICY_PATH)
        if policy_path.exists():
            logger.info(f"RL policy found at {policy_path} — running in RL mode.")
            results = _run_rl_mode(
                df,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                label_col,
                label_names,
                content_cols,
                num_classes,
                target_size,
                baseline,
                csv_path,
                out_dir,
                df,
                feature_mode=feature_mode,
                text_col=resolved_text_col,
                text_numeric_cols=text_numeric_cols,
                train_texts_for_novelty=train_texts_for_novelty,
                rows_to_X=rows_to_X,
            )
        else:
            logger.info("No RL policy found — running in bandit mode (fallback).")
            pool_size = int(target_size * getattr(cfg, "BANDIT_POOL_MULTIPLIER", 3))
            logger.info(
                f"Pre-generating synthetic pool: {pool_size} rows "
                f"(target={target_size} × multiplier={pool_size // max(target_size, 1)})"
            )
            synth_pool = generate_synthetic_pool(
                df=df,
                label_col=label_col,
                label_names=label_names,
                content_cols=content_cols,
                target_size=pool_size,
            )
            results = _run_bandit_mode(
                synth_pool,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                label_col,
                label_names,
                content_cols,
                num_classes,
                target_size,
                baseline,
                csv_path,
                out_dir,
                df,
                feature_mode=feature_mode,
                text_col=resolved_text_col,
                text_numeric_cols=text_numeric_cols,
                train_texts_for_novelty=train_texts_for_novelty,
                rows_to_X=rows_to_X,
                rows_to_y=rows_to_y,
            )

        results["classifier_model"] = clf
        results["augmentation_mode"] = mode
        return results

    finally:
        cfg.CLASSIFIER_MODEL = _orig_clf
        cfg.RL_MAX_STEPS = _orig_rl_max_steps
        cfg.EARLY_STOP_PATIENCE = _orig_patience
        if _orig_curation_threshold is None:
            try:
                del cfg.CURATION_THRESHOLD
            except AttributeError:
                pass
        else:
            cfg.CURATION_THRESHOLD = _orig_curation_threshold


# ─────────────────────────────────────────────────────────────────────
# RL MODE — agent decides everything from scratch, zero pre-generation
# ─────────────────────────────────────────────────────────────────────


def _run_rl_mode(
    df,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    label_col,
    label_names,
    content_cols,
    num_classes,
    target_size,
    baseline,
    csv_path,
    out_dir,
    original_df,
    *,
    feature_mode: str,
    text_col: str | None,
    text_numeric_cols: list[str],
    train_texts_for_novelty,
    rows_to_X,
):
    from agent import RLAgent

    def _rl_generate(df, label_col, label_names, content_cols, target_size, **_kw):
        # RL agent requests exact batch sizes (10 / 50); do not apply the
        # bandit-mode pool multiplier here.
        return generate_synthetic_pool(
            df=df,
            label_col=label_col,
            label_names=label_names,
            content_cols=content_cols,
            target_size=target_size,
            pool_multiplier=1,
        )

    rl_budget = cfg.rl_effective_budget(len(X_train), target_size)
    logger.info(
        f"RL budget (effective) = {rl_budget} "
        f"(n_real={len(X_train)}, target={target_size}, "
        f"scaled={cfg.RL_BUDGET_SCALE_WITH_DATA})"
    )

    agent = RLAgent(
        policy_path=cfg.RL_POLICY_PATH,
        max_steps=cfg.RL_MAX_STEPS,
        vecnormalize_path=cfg.RL_VECNORMALIZE_PATH,
    )
    episode = agent.run_episode(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        df=df,
        label_col=label_col,
        label_names=label_names,
        content_cols=content_cols,
        num_classes=num_classes,
        build_model_fn=build_model,
        train_fn=train,
        evaluate_f1_fn=evaluate_f1,
        get_probs_fn=get_probabilities,
        generate_fn=_rl_generate,
        bandit_filter_fn=functools.partial(
            _bandit_filter,
            X_train=X_train,
            feature_mode=feature_mode,
            text_col=text_col,
            text_numeric_cols=text_numeric_cols,
            train_texts=train_texts_for_novelty,
        ),
        target_size=target_size,
        total_budget=rl_budget,
        rows_to_x_batch=rows_to_X,
    )

    bandit_summary = dict(episode.get("bandit_info") or {})
    bandit_summary["agent"] = "ppo_rl"
    bandit_summary["step_log"] = episode["step_log"]
    bandit_summary["budget_used"] = episode["budget_used"]
    bandit_summary["rl_total_budget"] = episode.get("total_budget", rl_budget)
    bandit_summary["n_retrains"] = episode["n_retrains"]
    bandit_summary["feature_mode"] = feature_mode
    bandit_summary["text_col"] = text_col

    results = {
        "mode": "rl_agent",
        "dataset": str(csv_path),
        "feature_mode": feature_mode,
        "text_col": text_col,
        "text_numeric_cols": text_numeric_cols,
        "bandit_guided": {
            "test_f1": episode["final_f1"],
            "bandit_summary": bandit_summary,
        },
        "rl_guided": {
            "test_f1": episode["final_f1"],
            "baseline_f1": episode["baseline_f1"],
            "f1_gain": episode["f1_gain"],
            "curated_rows": len(episode["curated"]),
            "budget_used": episode["budget_used"],
            "total_budget": episode.get("total_budget", rl_budget),
            "effective_budget": rl_budget,
            "n_retrains": episode["n_retrains"],
            "step_log": episode["step_log"],
            "bandit_info": episode["bandit_info"],
        },
        "baseline_real_only": baseline,
    }

    save_outputs(episode["curated"], original_df, results, out_dir)
    return results


# ─────────────────────────────────────────────────────────────────────
# BANDIT MODE — pre-generate pool then filter (original behaviour)
# ─────────────────────────────────────────────────────────────────────


def _run_bandit_mode(
    synth_pool,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    label_col,
    label_names,
    content_cols,
    num_classes,
    target_size,
    baseline,
    csv_path,
    out_dir,
    original_df,
    *,
    feature_mode: str,
    text_col: str | None,
    text_numeric_cols: list[str],
    train_texts_for_novelty,
    rows_to_X,
    rows_to_y,
):
    model, bandit = (
        build_model(num_classes),
        ThompsonBandit(context_boost=cfg.THOMPSON_CONTEXT_BOOST),
    )
    train(model, (X_train, y_train))
    prev_f1 = evaluate_f1(model, (X_val, y_val))
    curated = []
    batch_rows, batch_actions = [], []  # batch_actions: (action, is_warm, ver)

    try:
        from features.novelty import novelty_scores_for_synthetic_vs_train

        pool_novelty = novelty_scores_for_synthetic_vs_train(
            X_train,
            content_cols,
            synth_pool,
            feature_mode=feature_mode,
            text_col=text_col,
            text_numeric_cols=text_numeric_cols,
            train_texts=train_texts_for_novelty,
        )
    except Exception as e:
        logger.warning(f"Novelty disabled for bandit pass (using 0.5): {e}")
        pool_novelty = np.full(len(synth_pool), 0.5, dtype=np.float64)

    X_pool = rows_to_X(synth_pool)

    for step, row in enumerate(synth_pool):
        x = X_pool[step : step + 1]
        y = _label_str_to_int(row[label_col], label_names)
        probs = get_probabilities(model, None, x)
        unc = compute_uncertainty(probs)[0]
        ver = compute_verifier_confidence(probs, [y])[0]
        novelty = float(pool_novelty[step])
        is_warm = step < cfg.BANDIT_WARM_START
        action = (
            1
            if is_warm
            else bandit.select(np.array([unc, novelty, ver], dtype=np.float64))
        )
        batch_rows.append(row)
        batch_actions.append((action, is_warm, float(ver)))

        if len(batch_rows) >= cfg.SYNTH_BATCH_SIZE:
            for r, (a, _, _v) in zip(batch_rows, batch_actions):
                if a == 1:
                    curated.append(r)

            if len(curated) > 0:
                temp = build_model(num_classes)
                train(
                    temp,
                    (
                        np.vstack([X_train, rows_to_X(curated)]),
                        np.concatenate([y_train, rows_to_y(curated)]),
                    ),
                )
                new_f1 = evaluate_f1(temp, (X_val, y_val))
                prev_f1 = new_f1
                model = temp

            for a, is_warm_step, v in batch_actions:
                if not is_warm_step:
                    if a == 1:
                        r = bandit.compute_accept_reward(v)
                        bandit.update(1, r)
                    else:
                        r = bandit.compute_reject_reward(v)
                        bandit.update(0, r)

            batch_rows, batch_actions = [], []

        if len(curated) >= target_size:
            logger.info(
                f"✔ Target reached: {len(curated)}/{target_size} curated rows — "
                "stopping bandit pass early."
            )
            break

    # Flush any remaining partial batch
    if batch_rows:
        for r, (a, _, _v) in zip(batch_rows, batch_actions):
            if a == 1:
                curated.append(r)
        for a, is_warm_step, v in batch_actions:
            if not is_warm_step:
                if a == 1:
                    bandit.update(1, bandit.compute_accept_reward(v))
                else:
                    bandit.update(0, bandit.compute_reject_reward(v))

    if curated:
        # FIX: naive baseline deferred to here — only run if curation produced
        # rows, avoiding a wasted train+eval when the pool is empty.
        naive = run_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            extra_X=rows_to_X(synth_pool),
            extra_y=rows_to_y(synth_pool),
            label="Naive aug (all synthetic, unfiltered)",
        )
        fm = build_model(num_classes)
        train(
            fm,
            (
                np.vstack([X_train, rows_to_X(curated)]),
                np.concatenate([y_train, rows_to_y(curated)]),
            ),
        )
        final_f1 = evaluate_f1(fm, (X_test, y_test))
    else:
        logger.warning("No samples curated — returning baseline.")
        naive = baseline
        final_f1 = baseline["test_f1"]

    logger.info(f"Final Test F1: {final_f1:.4f}")
    bandit_summary = bandit.summary()
    bandit_summary["novelty_mean_pool"] = (
        float(np.mean(pool_novelty)) if len(pool_novelty) else None
    )
    bandit_summary["feature_mode"] = feature_mode
    bandit_summary["text_col"] = text_col
    results = {
        "mode": "bandit",
        "dataset": str(csv_path),
        "feature_mode": feature_mode,
        "text_col": text_col,
        "text_numeric_cols": text_numeric_cols,
        "bandit_guided": {"test_f1": float(final_f1), "bandit_summary": bandit_summary},
        "baseline_real_only": baseline,
        "naive_augmentation": naive,
    }
    save_outputs(curated, original_df, results, out_dir)
    return results
