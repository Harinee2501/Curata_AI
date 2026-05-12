# config.py
# ─────────────────────────────────────────────────────────────────────
# Loads configuration from:
#   - .env               → sensitive values (API keys)
#   - config.yaml        → all other settings
# ─────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


def _load_yaml() -> dict:
    yaml_path = Path(__file__).parent / "config.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {yaml_path}")
    with open(yaml_path) as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self):
        raw = _load_yaml()

        # ── LLM (from environment) ───────────────────────────────────
        self.LLM_API_KEY:   str = os.getenv("LLM_API_KEY", "")
        self.LLM_MODEL:     str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

        # ── Classifier ───────────────────────────────────────────────
        self.CLASSIFIER_MODEL: str   = raw["classifier_model"]
        self.MAX_SEQ_LEN:      int   = raw["max_seq_len"]
        self.TRAIN_EPOCHS:     int   = raw["train_epochs"]
        self.TRAIN_BATCH_SIZE: int   = raw["train_batch_size"]
        self.LEARNING_RATE: float = float(raw["learning_rate"])

        # ── Embeddings ───────────────────────────────────────────────
        self.EMBEDDING_MODEL: str = raw["embedding_model"]

        # ── Synthetic generation ─────────────────────────────────────
        self.SYNTH_POOL_MULTIPLIER: int   = raw["synth_pool_multiplier"]
        self.SYNTH_BATCH_SIZE:      int   = raw["synth_batch_size"]
        self.LLM_TEMPERATURE:       float = raw["llm_temperature"]
        self.LLM_MAX_TOKENS:        int   = raw["llm_max_tokens"]
        self.LLM_MAX_RETRIES:       int   = raw["llm_max_retries"]
        self.LLM_RETRY_WAIT:        int   = raw["llm_retry_wait"]

        # ── Pipeline ─────────────────────────────────────────────────
        self.NUM_ITERATIONS:      int   = raw["num_iterations"]
        self.RANDOM_SEED:         int   = raw["random_seed"]
        self.MIN_REAL_SAMPLES:    int   = raw["min_real_samples"]
        self.EARLY_STOP_PATIENCE: int   = raw["early_stop_patience"]
        self.EARLY_STOP_DELTA:    float = raw["early_stop_delta"]

        # ── Data splits ──────────────────────────────────────────────
        self.VAL_RATIO:  float = raw["val_ratio"]
        self.TEST_RATIO: float = raw["test_ratio"]

        # ── Outputs ──────────────────────────────────────────────────
        self.OUTPUT_DIR: str = raw["output_dir"]

    def validate_llm(self):
        if not self.LLM_API_KEY:
            raise EnvironmentError(
                "\n\n[config] LLM_API_KEY is not set.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Add your Groq API key (free at https://console.groq.com)\n"
            )
        logger.info(f"LLM: {self.LLM_MODEL} via Groq")


cfg = Config()
