# pipeline.py (IMPROVED BANDIT VERSION)

import sys
import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))

from config import cfg
from data.dataset_utils import load_user_csv
from data.generate_synthetic import generate_synthetic_pool
from classifier.train import build_model, train
from classifier.evaluate import evaluate_f1, get_probabilities
from features.uncertainty import compute_uncertainty
from features.verifier import compute_verifier_confidence
from features.novelty import embed, compute_novelty
from bandit.thompson import ThompsonBandit


BATCH_SIZE  = 15        # ✅ larger batch → less noisy reward signal
WARM_START  = 5         # ✅ shorter warm start → bandit learns sooner

# ── Tuning knobs ───────────────────────────────────────────────────────────
# CONTEXT_BOOST     : weight of quality nudge on accept arm (default 0.3).
#                     Increase → accept arm is more aggressively favoured
#                     when quality is high. Decrease → bandit decides freely.
# REWARD_SCALE      : multiplier on delta-F1 before clipping (default 20).
#                     Higher → amplifies tiny F1 changes into bigger signals.
#                     Lower → dampens noise but may miss small improvements.
# REWARD_CLIP       : symmetric clip boundary on scaled reward (default 1.0).
# ──────────────────────────────────────────────────────────────────────────
CONTEXT_BOOST = 0.3
REWARD_SCALE  = 20.0
REWARD_CLIP   = 1.0


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--target", type=int, default=200)
    return parser.parse_args()


def _label_str_to_int(label_str, label_names):
    return {l: i for i, l in enumerate(label_names)}.get(str(label_str), 0)


def run_baseline(X_train, y_train, X_val, y_val, X_test, y_test, extra_X=None, extra_y=None, label="Baseline"):

    if extra_X is not None:
        X_train = np.vstack([X_train, extra_X])
        y_train = np.concatenate([y_train, extra_y])

    model = build_model(cfg.NUM_CLASSES)
    logger.info(f"── {label} ──")

    train(model, (X_train, y_train))

    val_f1 = evaluate_f1(model, (X_val, y_val))
    test_f1 = evaluate_f1(model, (X_test, y_test))

    logger.info(f"Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    return {"val_f1": val_f1, "test_f1": test_f1}


def save_outputs(curated_rows, original_df, results, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    curated_df = pd.DataFrame(curated_rows)
    curated_df.to_csv(out / "curated.csv", index=False)

    augmented = pd.concat([original_df, curated_df])
    augmented.to_csv(out / "augmented.csv", index=False)

    with open(out / "report.json", "w") as f:
        json.dump(results, f, indent=2)


def run_pipeline(csv_path, label_col, target_size):

    setup_logging(cfg.OUTPUT_DIR)

    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    logger.info("Loading dataset...")

    (
        df,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_names, num_classes, content_cols,
    ) = load_user_csv(csv_path, label_col)

    cfg.NUM_CLASSES = num_classes

    logger.info("Generating synthetic data...")
    synth_pool = generate_synthetic_pool(
        df=df,
        label_col=label_col,
        label_names=label_names,
        content_cols=content_cols,
        target_size=target_size
    )

    def rows_to_X(rows):
        return np.array([[float(r[col]) for col in content_cols] for r in rows])

    def rows_to_y(rows):
        return np.array([_label_str_to_int(r[label_col], label_names) for r in rows])

    # ── Baseline ─────────────────────────────
    baseline = run_baseline(X_train, y_train, X_val, y_val, X_test, y_test)

    # ── Naive ─────────────────────────────
    naive = run_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test,
        extra_X=rows_to_X(synth_pool),
        extra_y=rows_to_y(synth_pool),
        label="Naive Aug"
    )

    # ── Bandit (FIXED) ──────────────────────────────────────────────────────
    bandit = ThompsonBandit()
    curated = []

    model = build_model(cfg.NUM_CLASSES)
    train(model, (X_train, y_train))
    prev_f1 = evaluate_f1(model, (X_val, y_val))

    # Fix 4: embed real training samples ONCE so novelty is real, not 0.5
    real_texts = [" ".join(str(df.iloc[i][c]) for c in content_cols)
                  for i in range(len(X_train))]
    real_embeddings = embed(real_texts)
    logger.info(f"Embedded {len(real_texts)} real samples for novelty computation")

    batch_rows    = []
    batch_actions = []

    for step, row in enumerate(synth_pool):

        x = np.array([[float(row[col]) for col in content_cols]])
        y = _label_str_to_int(row[label_col], label_names)

        probs = get_probabilities(model, None, x)

        uncertainty = compute_uncertainty(probs)[0]
        verifier    = compute_verifier_confidence(probs, [y])[0]

        # Fix 4: compute real novelty via cosine distance from nearest real sample
        synth_text      = " ".join(str(row[c]) for c in content_cols)
        synth_embedding = embed([synth_text])
        novelty_score   = float(compute_novelty(real_embeddings, synth_embedding)[0])

        context = np.array([uncertainty, novelty_score, verifier])

        # Fix 5: WARM START forces accept but does NOT update bandit posterior
        if step < WARM_START:
            action = 1
            is_warm = True
        else:
            action   = bandit.select(context)
            is_warm  = False

        batch_rows.append(row)
        batch_actions.append((action, is_warm))

        if len(batch_rows) >= BATCH_SIZE:

            old_f1 = prev_f1

            # apply accepted samples
            for r, (a, _) in zip(batch_rows, batch_actions):
                if a == 1:
                    curated.append(r)

            if len(curated) > 0:
                curated_X = rows_to_X(curated)
                curated_y = rows_to_y(curated)

                temp_model = build_model(cfg.NUM_CLASSES)
                train(temp_model, (np.vstack([X_train, curated_X]), np.concatenate([y_train, curated_y])))

                new_f1 = evaluate_f1(temp_model, (X_val, y_val))

                # Fix 3: continuous delta-F1 reward — amplifies tiny changes, penalises regressions
                delta_f1 = new_f1 - old_f1
                reward   = float(np.clip(delta_f1 * REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))

                prev_f1 = new_f1
                model   = temp_model
            else:
                reward = 0.0

            # Fix 2 + Fix 5: update ONLY the arm that was chosen, skip warm-start steps
            for a, is_warm_step in batch_actions:
                if not is_warm_step:
                    bandit.update(a, reward)

            batch_rows    = []
            batch_actions = []

        if len(curated) >= target_size:
            break

    # ── handle last partial batch ────────────────────────────────────────────
    if len(batch_rows) > 0:
        for a, is_warm_step in batch_actions:
            if not is_warm_step:
                bandit.update(a, 0.0)

    # ── Final training ─────────────────────
    curated_X = rows_to_X(curated)
    curated_y = rows_to_y(curated)

    final_model = build_model(cfg.NUM_CLASSES)
    train(final_model, (np.vstack([X_train, curated_X]), np.concatenate([y_train, curated_y])))

    final_f1 = evaluate_f1(final_model, (X_test, y_test))

    logger.info(f"Final Test F1: {final_f1:.4f}")

    results = {
        "dataset": str(csv_path),
        "bandit_guided": {
            "test_f1": float(final_f1),
            "bandit_summary": bandit.summary(),
        },
        "baseline_real_only": baseline,
        "naive_augmentation": naive,
    }

    save_outputs(curated, df, results, cfg.OUTPUT_DIR)
    return results


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        csv_path=args.csv,
        label_col=args.label_col,
        target_size=args.target
    )