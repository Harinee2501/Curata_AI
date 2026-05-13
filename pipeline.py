# pipeline.py — RL-integrated curation pipeline
#
# TWO MODES:
#   1. RL mode    — agent starts from ZERO rows and decides everything:
#                   when to generate, how much, when to filter,
#                   when to retrain, and when to stop.
#                   No pre-generation. No naive baseline run upfront.
#
#   2. Bandit mode — original Thompson bandit loop (fallback).
#                   Pre-generates a pool, then filters it.
#
# Switch RL vs bandit by placing/removing the PPO zip named in config.yaml
# (default: best_model.zip) in the project root.

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

BATCH_SIZE = 10
WARM_START = 10


def setup_logging(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def _label_str_to_int(label_str, label_names):
    return {l: i for i, l in enumerate(label_names)}.get(str(label_str), 0)


def run_baseline(X_train, y_train, X_val, y_val, X_test, y_test,
                 extra_X=None, extra_y=None, label="Baseline"):
    if extra_X is not None:
        X_train = np.vstack([X_train, extra_X])
        y_train = np.concatenate([y_train, extra_y])
    model = build_model(cfg.NUM_CLASSES)
    logger.info(f"── {label} ──")
    train(model, (X_train, y_train))
    val_f1  = evaluate_f1(model, (X_val,  y_val))
    test_f1 = evaluate_f1(model, (X_test, y_test))
    logger.info(f"Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
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
):
    """
    Thompson accept/reject with context [uncertainty, novelty, verifier].
    Novelty uses sentence embeddings vs real training rows (see features.novelty).
    """
    bandit, accepted, verifier_scores = ThompsonBandit(), [], []
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
        x       = X_mat[i : i + 1]
        y       = _label_str_to_int(row[label_col], label_names)
        probs   = get_probs_fn(model, None, x)
        unc     = compute_uncertainty(probs)[0]
        ver     = compute_verifier_confidence(probs, [y])[0]
        novelty = float(novelty_vec[i])
        action = (
            1
            if i < WARM_START
            else bandit.select(np.array([unc, novelty, ver], dtype=np.float64))
        )
        if action == 1:
            accepted.append(row)
            verifier_scores.append(float(ver))
            novelties_accepted.append(novelty)
        bandit.update(action, 1 if ver > 0.6 else 0)
    acc_rate = len(accepted) / max(len(rows), 1)
    vc_mean  = float(np.mean(verifier_scores)) if verifier_scores else 0.5
    summary  = bandit.summary()
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

def run_pipeline(csv_path, label_col, target_size, output_dir=None, text_col=None):
    out_dir = output_dir or cfg.OUTPUT_DIR
    setup_logging(out_dir)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

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
    ) = load_user_csv(csv_path, label_col, text_col=text_col)
    cfg.NUM_CLASSES = num_classes

    def rows_to_X(rows):
        return rows_dicts_to_matrix(
            rows,
            feature_mode=feature_mode,
            content_cols=content_cols,
            text_col=resolved_text_col,
            text_numeric_cols=text_numeric_cols,
        )

    rows_to_y = lambda rows: np.array(
        [_label_str_to_int(r[label_col], label_names) for r in rows]
    )

    # Baseline on real data only — always needed for final comparison
    baseline = run_baseline(X_train, y_train, X_val, y_val, X_test, y_test,
                            label="Baseline — real only")

    policy_path = Path(cfg.RL_POLICY_PATH)
    if policy_path.exists():
        logger.info(f"RL policy found at {policy_path} — running in RL mode.")
        logger.info("Agent starts from zero synthetic rows — deciding everything autonomously.")
        return _run_rl_mode(
            df, X_train, y_train, X_val, y_val, X_test, y_test,
            label_col, label_names, content_cols, num_classes,
            target_size, baseline, csv_path, out_dir, df,
            feature_mode=feature_mode,
            text_col=resolved_text_col,
            text_numeric_cols=text_numeric_cols,
            train_texts_for_novelty=train_texts_for_novelty,
            rows_to_X=rows_to_X,
        )
    else:
        logger.info("No RL policy found — running in bandit mode (fallback).")
        logger.info("Pre-generating synthetic pool for bandit filtering...")
        synth_pool = generate_synthetic_pool(
            df=df, label_col=label_col, label_names=label_names,
            content_cols=content_cols, target_size=target_size,
        )
        naive = run_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            extra_X=rows_to_X(synth_pool), extra_y=rows_to_y(synth_pool),
            label="Naive Aug",
        )
        return _run_bandit_mode(
            synth_pool, X_train, y_train, X_val, y_val, X_test, y_test,
            label_col, label_names, content_cols, num_classes,
            target_size, baseline, naive, csv_path, out_dir, df,
            feature_mode=feature_mode,
            text_col=resolved_text_col,
            text_numeric_cols=text_numeric_cols,
            train_texts_for_novelty=train_texts_for_novelty,
            rows_to_X=rows_to_X,
        )


# ─────────────────────────────────────────────────────────────────────
# RL MODE — agent decides everything from scratch, zero pre-generation
# ─────────────────────────────────────────────────────────────────────

def _run_rl_mode(
    df, X_train, y_train, X_val, y_val, X_test, y_test,
    label_col, label_names, content_cols, num_classes,
    target_size, baseline, csv_path, out_dir, original_df,
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
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        df=df,
        label_col=label_col,
        label_names=label_names,
        content_cols=content_cols,
        num_classes=num_classes,
        build_model_fn   = build_model,
        train_fn         = train,
        evaluate_f1_fn   = evaluate_f1,
        get_probs_fn     = get_probabilities,
        generate_fn      = _rl_generate,
        bandit_filter_fn = functools.partial(
            _bandit_filter,
            X_train=X_train,
            feature_mode=feature_mode,
            text_col=text_col,
            text_numeric_cols=text_numeric_cols,
            train_texts=train_texts_for_novelty,
        ),
        target_size      = target_size,
        total_budget     = rl_budget,
        rows_to_x_batch  = rows_to_X,
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
        "mode":    "rl_agent",
        "dataset": str(csv_path),
        "feature_mode": feature_mode,
        "text_col": text_col,
        "text_numeric_cols": text_numeric_cols,
        # Same keys as bandit mode so Streamlit / FastAPI stay compatible.
        "bandit_guided": {
            "test_f1":       episode["final_f1"],
            "bandit_summary": bandit_summary,
        },
        "rl_guided": {
            "test_f1":      episode["final_f1"],
            "baseline_f1":  episode["baseline_f1"],
            "f1_gain":      episode["f1_gain"],
            "curated_rows": len(episode["curated"]),
            "budget_used":  episode["budget_used"],
            "total_budget": episode.get("total_budget", rl_budget),
            "effective_budget": rl_budget,
            "n_retrains":   episode["n_retrains"],
            "step_log":     episode["step_log"],
            "bandit_info":  episode["bandit_info"],
        },
        "baseline_real_only": baseline,
    }

    save_outputs(episode["curated"], original_df, results, out_dir)
    return results


# ─────────────────────────────────────────────────────────────────────
# BANDIT MODE — pre-generate pool then filter (original behaviour)
# ─────────────────────────────────────────────────────────────────────

def _run_bandit_mode(
    synth_pool, X_train, y_train, X_val, y_val, X_test, y_test,
    label_col, label_names, content_cols, num_classes,
    target_size, baseline, naive, csv_path, out_dir, original_df,
    *,
    feature_mode: str,
    text_col: str | None,
    text_numeric_cols: list[str],
    train_texts_for_novelty,
    rows_to_X,
):
    rows_to_y = lambda rows: np.array(
        [_label_str_to_int(r[label_col], label_names) for r in rows]
    )

    model, bandit         = build_model(num_classes), ThompsonBandit()
    train(model, (X_train, y_train))
    prev_f1               = evaluate_f1(model, (X_val, y_val))
    curated               = []
    batch_rows, batch_actions = [], []

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
        x      = X_pool[step : step + 1]
        y      = _label_str_to_int(row[label_col], label_names)
        probs  = get_probabilities(model, None, x)
        unc    = compute_uncertainty(probs)[0]
        ver    = compute_verifier_confidence(probs, [y])[0]
        novelty = float(pool_novelty[step])
        action = (
            1
            if step < WARM_START
            else bandit.select(np.array([unc, novelty, ver], dtype=np.float64))
        )
        batch_rows.append(row)
        batch_actions.append(action)

        if len(batch_rows) >= BATCH_SIZE:
            accepted = [r for r, a in zip(batch_rows, batch_actions) if a == 1]
            if accepted:
                curated.extend(accepted)
                temp = build_model(num_classes)
                train(temp, (np.vstack([X_train, rows_to_X(curated)]),
                             np.concatenate([y_train, rows_to_y(curated)])))
                new_f1  = evaluate_f1(temp, (X_val, y_val))
                reward  = 1 if new_f1 > prev_f1 else 0
                prev_f1 = new_f1
                model   = temp
                for a in batch_actions:
                    bandit.update(a, reward)
            batch_rows, batch_actions = [], []

        if len(curated) >= target_size:
            logger.info(
                f"✔ Target reached: {len(curated)}/{target_size} curated rows — "
                "stopping bandit pass early."
            )
            break

    if curated:
        fm = build_model(num_classes)
        train(fm, (np.vstack([X_train, rows_to_X(curated)]),
                   np.concatenate([y_train, rows_to_y(curated)])))
        final_f1 = evaluate_f1(fm, (X_test, y_test))
    else:
        logger.warning("No samples curated — returning baseline.")
        final_f1 = baseline["test_f1"]

    logger.info(f"Final Test F1: {final_f1:.4f}")
    bandit_summary = bandit.summary()
    bandit_summary["novelty_mean_pool"] = (
        float(np.mean(pool_novelty)) if len(pool_novelty) else None
    )
    bandit_summary["feature_mode"] = feature_mode
    bandit_summary["text_col"] = text_col
    results = {
        "mode":               "bandit",
        "dataset":            str(csv_path),
        "feature_mode":       feature_mode,
        "text_col":           text_col,
        "text_numeric_cols":  text_numeric_cols,
        "bandit_guided":      {"test_f1": float(final_f1), "bandit_summary": bandit_summary},
        "baseline_real_only": baseline,
        "naive_augmentation": naive,
    }
    save_outputs(curated, original_df, results, out_dir)
    return results