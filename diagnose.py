# diagnose.py
# ─────────────────────────────────────────────────────────────────────
# Drop this in your project root and run:
#   python diagnose.py your_data.csv label_column_name
#
# It prints:
#   1. Dataset shape and class distribution
#   2. Verifier confidence distribution on a small synthetic pool
#   3. How many rows each mode would accept at its threshold
#   4. What the bandit's hard_block cutoff is per mode
# ─────────────────────────────────────────────────────────────────────

import sys
import numpy as np
import pandas as pd

csv_path = sys.argv[1]
label_col = sys.argv[2]

print("\n" + "=" * 60)
print("CURATA DIAGNOSTICS")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────────────
from data.dataset_utils import load_user_csv

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
    _,
) = load_user_csv(csv_path, label_col)

print(f"\n[Dataset]")
print(f"  Total rows      : {len(df)}")
print(f"  Train rows      : {len(X_train)}")
print(f"  Val rows        : {len(X_val)}")
print(f"  Test rows       : {len(X_test)}")
print(f"  Classes         : {num_classes}  →  {label_names}")
print(f"  Feature mode    : {feature_mode}")
print(f"  Content cols    : {content_cols}")
print(f"  Class dist (train):")
for i, name in enumerate(label_names):
    count = int((y_train == i).sum())
    pct = 100 * count / max(len(y_train), 1)
    print(f"    [{i}] {name:20s}  {count:4d} rows  ({pct:.1f}%)")

# ── 2. Train baseline model ───────────────────────────────────────────
from classifier.train import build_model, train
from classifier.evaluate import evaluate_f1, get_probabilities
from features.uncertainty import compute_uncertainty
from features.verifier import compute_verifier_confidence
from data.row_features import rows_dicts_to_matrix

model = build_model(num_classes)
train(model, (X_train, y_train))
val_f1 = evaluate_f1(model, (X_val, y_val))
test_f1 = evaluate_f1(model, (X_test, y_test))
print(f"\n[Baseline model]")
print(f"  Val F1  : {val_f1:.4f}")
print(f"  Test F1 : {test_f1:.4f}")

# ── 3. Generate a small probe pool and score it ───────────────────────
from data.generate_synthetic import generate_synthetic_pool

PROBE_SIZE = 50
print(f"\n[Generating {PROBE_SIZE} synthetic rows for scoring...]")
try:
    probe_pool = generate_synthetic_pool(
        df=df,
        label_col=label_col,
        label_names=label_names,
        content_cols=content_cols,
        target_size=PROBE_SIZE,
        pool_multiplier=1,
    )
except Exception as e:
    print(f"  Generation failed: {e}")
    sys.exit(1)

print(f"  Generated : {len(probe_pool)} rows")

X_probe = rows_dicts_to_matrix(
    probe_pool,
    feature_mode=feature_mode,
    content_cols=content_cols,
    text_col=resolved_text_col,
    text_numeric_cols=text_numeric_cols,
)

ver_scores = []
unc_scores = []
for i, row in enumerate(probe_pool):
    x = X_probe[i : i + 1]
    y = {l: j for j, l in enumerate(label_names)}.get(str(row[label_col]), 0)
    probs = get_probabilities(model, None, x)
    unc = compute_uncertainty(probs)[0]
    ver = compute_verifier_confidence(probs, [y])[0]
    ver_scores.append(float(ver))
    unc_scores.append(float(unc))

ver_arr = np.array(ver_scores)
unc_arr = np.array(unc_scores)

print(f"\n[Verifier confidence — {len(ver_arr)} rows]")
print(f"  min    : {ver_arr.min():.4f}")
print(f"  p10    : {np.percentile(ver_arr, 10):.4f}")
print(f"  p25    : {np.percentile(ver_arr, 25):.4f}")
print(f"  median : {np.median(ver_arr):.4f}")
print(f"  p75    : {np.percentile(ver_arr, 75):.4f}")
print(f"  p90    : {np.percentile(ver_arr, 90):.4f}")
print(f"  max    : {ver_arr.max():.4f}")
print(f"  mean   : {ver_arr.mean():.4f}  ± {ver_arr.std():.4f}")

print(f"\n[Uncertainty — {len(unc_arr)} rows]")
print(f"  mean   : {unc_arr.mean():.4f}  ± {unc_arr.std():.4f}")
print(f"  median : {np.median(unc_arr):.4f}")

# ── 4. Per-mode acceptance simulation ────────────────────────────────
from config import cfg

print(f"\n[Mode acceptance simulation on {len(ver_arr)} probe rows]")
print(
    f"  {'Mode':<12} {'threshold':>10} {'hard_block':>11} {'accepted':>9} {'accept%':>8}"
)
print(f"  {'-' * 12} {'-' * 10} {'-' * 11} {'-' * 9} {'-' * 8}")

for mode in ("fast", "balanced", "thorough"):
    mc = cfg.get_mode_config(mode)
    threshold = float(mc["curation_threshold"])
    hard_block = max(0.0, threshold - 0.10)
    accepted = int((ver_arr >= hard_block).sum())
    pct = 100 * accepted / max(len(ver_arr), 1)
    print(
        f"  {mode:<12} {threshold:>10.2f} {hard_block:>11.2f} {accepted:>9d} {pct:>7.1f}%"
    )

# ── 5. What threshold would accept ~50–70% of rows? ──────────────────
for target_pct in (30, 50, 70):
    thr = float(np.percentile(ver_arr, 100 - target_pct))
    print(f"\n  To accept ~{target_pct}% of rows, threshold should be ≤ {thr:.3f}")

print("\n" + "=" * 60)
print("Paste this output when reporting issues.")
print("=" * 60 + "\n")
