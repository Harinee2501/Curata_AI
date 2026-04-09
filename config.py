# config.py
# -------------------------------------------------------------------
# Central config for the entire project.
# Change values here instead of hunting through individual files.
# -------------------------------------------------------------------

# ── Dataset ─────────────────────────────────────────────────────────
DATASET_NAME   = "imdb"          # HuggingFace dataset to use as "real" data
NUM_CLASSES    = 2               # number of classification labels
LABEL_NAMES    = ["negative", "positive"]

# How many real samples to simulate a "low-resource" scenario
REAL_TRAIN_SIZE = 500
VAL_SIZE        = 200
TEST_SIZE       = 200

# ── Synthetic Data ───────────────────────────────────────────────────
SYNTH_POOL_SIZE    = 1000        # total LLM-generated candidates
SYNTH_TARGET_SIZE  = 300         # how many we want to end up keeping
SYNTH_BATCH_SIZE   = 50          # how many synthetics to evaluate per iteration

# ── Classifier ──────────────────────────────────────────────────────
CLASSIFIER_MODEL = "distilbert-base-uncased"
MAX_SEQ_LEN      = 128
TRAIN_EPOCHS     = 2             # keep low so the pipeline runs fast
TRAIN_BATCH_SIZE = 16
LEARNING_RATE    = 2e-5

# ── Embeddings (for semantic novelty) ───────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast + good sentence embeddings

# ── Bandit ──────────────────────────────────────────────────────────
BANDIT_ALGO  = "ucb"    # "ucb" or "thompson"
UCB_ALPHA    = 1.0      # exploration constant for UCB

# ── Pipeline ────────────────────────────────────────────────────────
NUM_ITERATIONS = 20     # max bandit rounds (stops early if pool fills up)
RANDOM_SEED    = 42

# ── Paths ────────────────────────────────────────────────────────────
SYNTH_DATA_PATH  = "data/synthetic_pool.json"
RESULTS_PATH     = "results.json"
