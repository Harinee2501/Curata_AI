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

        # ── RL agent (PPO policy) ────────────────────────────────────
        # If `rl_policy_path` exists under the project directory, pipeline runs
        # in autonomous RL mode (see agent.py). Otherwise bandit-only mode.
        _rl_rel = raw.get("rl_policy_path", "best_model.zip")
        _rl_path = Path(_rl_rel)
        self.RL_POLICY_PATH: str = (
            str(_rl_path.resolve())
            if _rl_path.is_absolute()
            else str((Path(__file__).parent / _rl_rel).resolve())
        )
        _vn = raw.get("rl_vecnormalize_path") or ""
        if isinstance(_vn, str) and _vn.strip():
            _vn_path = Path(_vn.strip())
            self.RL_VECNORMALIZE_PATH: str | None = (
                str(_vn_path.resolve())
                if _vn_path.is_absolute()
                else str((Path(__file__).parent / _vn.strip()).resolve())
            )
        else:
            self.RL_VECNORMALIZE_PATH = None
        self.RL_MAX_STEPS: int = int(raw.get("rl_max_steps", 40))
        self.RL_TOTAL_BUDGET: int = int(raw.get("rl_total_budget", 300))
        self.RL_BUDGET_SCALE_WITH_DATA: bool = bool(
            raw.get("rl_budget_scale_with_data", True)
        )
        self.RL_BUDGET_BASE: float = float(raw.get("rl_budget_base", 80))
        self.RL_BUDGET_PER_TARGET_ROW: float = float(
            raw.get("rl_budget_per_target_row", 1.2)
        )
        self.RL_BUDGET_PER_REAL_ROW: float = float(
            raw.get("rl_budget_per_real_row", 0.35)
        )
        self.RL_BUDGET_MIN: int = int(raw.get("rl_budget_min", 120))
        self.RL_BUDGET_MAX: int = int(raw.get("rl_budget_max", 900))

    def rl_effective_budget(self, n_real_rows: int, target_size: int) -> int:
        """
        RL action budget (generate/filter/retrain costs). Scales with dataset
        and target when rl_budget_scale_with_data is true.
        """
        if not self.RL_BUDGET_SCALE_WITH_DATA:
            return int(self.RL_TOTAL_BUDGET)
        n_real = max(int(n_real_rows), 0)
        tgt = max(int(target_size), 1)
        raw_budget = (
            self.RL_BUDGET_BASE
            + self.RL_BUDGET_PER_TARGET_ROW * tgt
            + self.RL_BUDGET_PER_REAL_ROW * n_real
        )
        return int(
            round(
                min(
                    self.RL_BUDGET_MAX,
                    max(self.RL_BUDGET_MIN, raw_budget),
                )
            )
        )

    def validate_llm(self):
        if not self.LLM_API_KEY:
            raise EnvironmentError(
                "\n\n[config] LLM_API_KEY is not set.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Add your Groq API key (free at https://console.groq.com)\n"
            )
        logger.info(f"LLM: {self.LLM_MODEL} via Groq")


cfg = Config()