# pipeline.py
# -------------------------------------------------------------------
# Main closed-loop pipeline for RL-Guided Synthetic Data Selection.
#
# Each iteration:
#   1. Take a batch from the synthetic pool
#   2. Extract 3 quality features per sample
#   3. Bandit scores each sample → accept / reject
#   4. Retrain classifier on real + accepted data
#   5. Compute ΔF1 as reward
#   6. Update bandit policy
#   Repeat until 300 curated samples are collected.
# -------------------------------------------------------------------

import sys, os, json, random
import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.dataset_utils     import load_real_data, TextClassificationDataset
from data.generate_synthetic import generate_synthetic_pool
from classifier.train       import get_tokenizer, build_model, train
from classifier.evaluate    import evaluate_f1, get_probabilities
from features.uncertainty   import compute_uncertainty
from features.novelty       import embed, compute_novelty
from features.verifier      import compute_verifier_confidence
from bandit.ucb             import UCBBandit
from bandit.thompson        import ThompsonBandit


# ── Helpers ─────────────────────────────────────────────────────────

def select_bandit(algo: str):
    if algo == "ucb":
        print("[pipeline] Using UCB bandit")
        return UCBBandit()
    elif algo == "thompson":
        print("[pipeline] Using Thompson Sampling bandit")
        return ThompsonBandit()
    else:
        raise ValueError(f"Unknown bandit algo '{algo}'. Choose 'ucb' or 'thompson'.")


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] Device: {device}")
    return device


# ── Baseline Evaluations ────────────────────────────────────────────

def run_baseline(train_texts, train_labels, val_texts, val_labels,
                 test_texts, test_labels, tokenizer, device,
                 extra_texts=None, extra_labels=None, label="Baseline"):
    """
    Trains and evaluates the classifier for one condition.
    Used to compute real-only and naive-augmentation baselines.
    """
    all_texts  = train_texts  + (extra_texts  or [])
    all_labels = train_labels + (extra_labels or [])

    train_ds = TextClassificationDataset(all_texts, all_labels, tokenizer)
    val_ds   = TextClassificationDataset(val_texts,  val_labels,  tokenizer)
    test_ds  = TextClassificationDataset(test_texts, test_labels, tokenizer)

    model = build_model()
    print(f"\n── {label} ──")
    train(model, train_ds, device)

    val_f1  = evaluate_f1(model, val_ds,  device)
    test_f1 = evaluate_f1(model, test_ds, device)
    print(f"  Val F1: {val_f1:.4f}  |  Test F1: {test_f1:.4f}")
    return {"val_f1": val_f1, "test_f1": test_f1}


# ── Main Pipeline ───────────────────────────────────────────────────

def run_pipeline():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    device    = get_device()
    tokenizer = get_tokenizer()

    # ── Step 1: Load data ──────────────────────────────────────────
    print("\n=== Loading Data ===")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_real_data()

    synth_pool = generate_synthetic_pool()   # list of {"text": ..., "label": ...}
    synth_pool = synth_pool[:]               # copy so we can pop from it
    random.shuffle(synth_pool)

    # ── Step 2: Pre-compute real embeddings for novelty signal ─────
    print("\n=== Pre-computing Real Data Embeddings ===")
    real_embeddings = embed(train_texts)     # shape (500, embedding_dim)

    # ── Step 3: Baseline — real only ──────────────────────────────
    print("\n=== Baseline: Real Data Only ===")
    baseline_results = run_baseline(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, tokenizer, device,
        label="Real Only"
    )

    # ── Step 4: Naive augmentation — all synthetic ────────────────
    print("\n=== Naive Augmentation: Real + All Synthetic ===")
    all_synth_texts  = [s["text"]  for s in synth_pool]
    all_synth_labels = [s["label"] for s in synth_pool]
    naive_results = run_baseline(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, tokenizer, device,
        extra_texts=all_synth_texts, extra_labels=all_synth_labels,
        label="Naive Augmentation"
    )

    # ── Step 5: Bandit-guided selection ───────────────────────────
    print("\n=== Bandit-Guided Synthetic Data Selection ===")
    bandit         = select_bandit(config.BANDIT_ALGO)
    curated_texts  = []
    curated_labels = []
    reward_history = []          # track ΔF1 per iteration
    val_f1_history = []

    # Initialise with a model trained on real data only
    train_ds  = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_ds    = TextClassificationDataset(val_texts,   val_labels,   tokenizer)
    test_ds   = TextClassificationDataset(test_texts,  test_labels,  tokenizer)

    model     = build_model()
    train(model, train_ds, device)
    prev_f1   = evaluate_f1(model, val_ds, device)
    val_f1_history.append(prev_f1)
    print(f"  [iteration 0] Initial val F1: {prev_f1:.4f}")

    synth_idx = 0   # pointer into the synthetic pool

    for iteration in range(1, config.NUM_ITERATIONS + 1):
        # Stop if we've hit our target pool size
        if len(curated_texts) >= config.SYNTH_TARGET_SIZE:
            print(f"\n[pipeline] Target pool size {config.SYNTH_TARGET_SIZE} reached. Done.")
            break

        # ── Take next batch from synthetic pool ───────────────────
        batch_end  = min(synth_idx + config.SYNTH_BATCH_SIZE, len(synth_pool))
        batch      = synth_pool[synth_idx:batch_end]
        synth_idx  = batch_end

        if not batch:
            print("[pipeline] Synthetic pool exhausted.")
            break

        batch_texts  = [s["text"]  for s in batch]
        batch_labels = [s["label"] for s in batch]

        # ── Feature extraction ────────────────────────────────────
        probs             = get_probabilities(model, tokenizer, batch_texts, device)
        uncertainty       = compute_uncertainty(probs)

        synth_embeddings  = embed(batch_texts)
        novelty           = compute_novelty(real_embeddings, synth_embeddings)

        verifier_conf     = compute_verifier_confidence(probs, batch_labels)

        # ── Bandit selection ──────────────────────────────────────
        accepted_texts, accepted_labels = [], []

        for i in range(len(batch)):
            context = np.array([uncertainty[i], novelty[i], verifier_conf[i]])
            action  = bandit.select(context)

            if action == 1:   # accept
                accepted_texts.append(batch_texts[i])
                accepted_labels.append(batch_labels[i])

        curated_texts.extend(accepted_texts)
        curated_labels.extend(accepted_labels)

        # Cap at target size
        if len(curated_texts) > config.SYNTH_TARGET_SIZE:
            curated_texts  = curated_texts[:config.SYNTH_TARGET_SIZE]
            curated_labels = curated_labels[:config.SYNTH_TARGET_SIZE]

        # ── Retrain classifier ────────────────────────────────────
        merged_texts  = train_texts  + curated_texts
        merged_labels = train_labels + curated_labels
        merged_ds     = TextClassificationDataset(merged_texts, merged_labels, tokenizer)

        model = build_model()   # fresh model to avoid compounding bias
        train(model, merged_ds, device)

        # ── Reward computation ────────────────────────────────────
        curr_f1 = evaluate_f1(model, val_ds, device)
        reward  = curr_f1 - prev_f1
        prev_f1 = curr_f1

        reward_history.append(reward)
        val_f1_history.append(curr_f1)

        # ── Bandit policy update ──────────────────────────────────
        # Use the last arm that was most commonly pulled in this batch
        dominant_arm = 1 if accepted_texts else 0
        bandit.update(dominant_arm, reward)

        print(
            f"  [iteration {iteration:2d}] "
            f"accepted: {len(accepted_texts):3d}/{len(batch)} | "
            f"curated total: {len(curated_texts):3d} | "
            f"val F1: {curr_f1:.4f} | "
            f"ΔF1 (reward): {reward:+.4f}"
        )

    # ── Final evaluation ──────────────────────────────────────────
    print("\n=== Final Evaluation on Test Set ===")
    bandit_test_f1 = evaluate_f1(model, test_ds, device)
    print(f"  Bandit-Guided Test F1: {bandit_test_f1:.4f}")

    # ── Summary ───────────────────────────────────────────────────
    results = {
        "baseline_real_only":      baseline_results,
        "naive_augmentation":      naive_results,
        "bandit_guided": {
            "val_f1_history": val_f1_history,
            "reward_history": reward_history,
            "curated_pool_size": len(curated_texts),
            "test_f1": bandit_test_f1,
        },
    }

    with open(config.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[pipeline] Results saved to '{config.RESULTS_PATH}'")
    _print_summary(results)
    return results


def _print_summary(results):
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Real Only           Test F1: {results['baseline_real_only']['test_f1']:.4f}")
    print(f"  Naive Augmentation  Test F1: {results['naive_augmentation']['test_f1']:.4f}")
    print(f"  Bandit-Guided       Test F1: {results['bandit_guided']['test_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
