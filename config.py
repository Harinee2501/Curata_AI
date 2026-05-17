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

# All classifiers supported by classifier/train.py.
# xgboost and lightgbm require the respective packages to be installed.
VALID_CLASSIFIERS = frozenset(
    {
        "logistic_regression",
        "naive_bayes",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm",
    }
)


def _load_yaml() -> dict:
    yaml_path = Path(__file__).parent / "config.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {yaml_path}")
    with open(yaml_path) as f:
        return yaml.safe_load(f)


# ── Named augmentation mode presets ──────────────────────────────────
# Fallback defaults used if config.yaml is missing the section.
_DEFAULT_MODE_CONFIGS: dict[str, dict] = {
    "fast": {
        "pool_multiplier": 1,
        "rl_max_steps": 10,
        "patience": 3,
        "curation_threshold": 0.50,
    },
    "balanced": {
        "pool_multiplier": 3,
        "rl_max_steps": 40,
        "patience": 8,
        "curation_threshold": 0.65,
    },
    "thorough": {
        "pool_multiplier": 5,
        "rl_max_steps": 100,
        "patience": 20,
        "curation_threshold": 0.75,
    },
}

VALID_MODES = frozenset(_DEFAULT_MODE_CONFIGS)


class Config:
    def __init__(self):
        raw = _load_yaml()

        # ── LLM (from environment) ───────────────────────────────────
        self.LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

        # ── Classifier ───────────────────────────────────────────────
        classifier_model = str(raw["classifier_model"]).lower().strip()
        if classifier_model not in VALID_CLASSIFIERS:
            raise ValueError(
                f"[config] Unknown classifier_model '{classifier_model}'. "
                f"Choose one of: {', '.join(sorted(VALID_CLASSIFIERS))}"
            )
        self.CLASSIFIER_MODEL: str = classifier_model
        self.MAX_SEQ_LEN: int = raw["max_seq_len"]
        self.TRAIN_EPOCHS: int = raw["train_epochs"]
        self.TRAIN_BATCH_SIZE: int = raw["train_batch_size"]
        self.LEARNING_RATE: float = float(raw["learning_rate"])

        # ── Embeddings ───────────────────────────────────────────────
        self.EMBEDDING_MODEL: str = raw["embedding_model"]

        # ── Synthetic generation ─────────────────────────────────────
        self.SYNTH_POOL_MULTIPLIER: int = raw["synth_pool_multiplier"]
        self.SYNTH_BATCH_SIZE: int = raw["synth_batch_size"]
        self.LLM_TEMPERATURE: float = raw["llm_temperature"]
        self.LLM_MAX_TOKENS: int = raw["llm_max_tokens"]
        self.LLM_MAX_RETRIES: int = raw["llm_max_retries"]
        self.LLM_RETRY_WAIT: int = raw["llm_retry_wait"]

        # ── Thompson bandit ──────────────────────────────────────────
        self.BANDIT_WARM_START: int = int(raw.get("bandit_warm_start", 5))
        self.THOMPSON_CONTEXT_BOOST: float = float(
            raw.get("thompson_context_boost", 0.3)
        )
        self.THOMPSON_REWARD_SCALE: float = float(
            raw.get("thompson_reward_scale", 20.0)
        )
        self.THOMPSON_REWARD_CLIP: float = float(raw.get("thompson_reward_clip", 1.0))

        # ── Pipeline ─────────────────────────────────────────────────
        self.NUM_ITERATIONS: int = raw["num_iterations"]
        self.RANDOM_SEED: int = raw["random_seed"]
        self.MIN_REAL_SAMPLES: int = raw["min_real_samples"]
        self.EARLY_STOP_PATIENCE: int = raw["early_stop_patience"]
        self.EARLY_STOP_DELTA: float = raw["early_stop_delta"]

        # ── Data splits ──────────────────────────────────────────────
        self.VAL_RATIO: float = raw["val_ratio"]
        self.TEST_RATIO: float = raw["test_ratio"]

        # ── Outputs ──────────────────────────────────────────────────
        self.OUTPUT_DIR: str = raw["output_dir"]

        # ── Augmentation modes ───────────────────────────────────────
        # Load named mode presets from yaml; fall back to hardcoded defaults.
        yaml_modes: dict = raw.get("augmentation_modes") or {}
        self.AUGMENTATION_MODES: dict[str, dict] = {
            mode: {**_DEFAULT_MODE_CONFIGS.get(mode, {}), **yaml_modes.get(mode, {})}
            for mode in _DEFAULT_MODE_CONFIGS
        }
        self.DEFAULT_AUGMENTATION_MODE: str = str(
            raw.get("default_augmentation_mode", "balanced")
        )

        # ── RL agent (PPO policy) ────────────────────────────────────
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

    # ── Mode helpers ─────────────────────────────────────────────────

    def get_mode_config(self, mode: str | None = None) -> dict:
        """
        Return the preset dict for *mode* (one of 'fast', 'balanced', 'thorough').
        Falls back to the configured default when *mode* is None or unrecognised.
        """
        m = (mode or self.DEFAULT_AUGMENTATION_MODE).lower().strip()
        if m not in self.AUGMENTATION_MODES:
            logger.warning(
                f"[config] Unknown augmentation mode '{m}' — "
                f"falling back to '{self.DEFAULT_AUGMENTATION_MODE}'."
            )
            m = self.DEFAULT_AUGMENTATION_MODE
        return dict(self.AUGMENTATION_MODES[m])

    def pool_size_for_mode(self, mode: str | None, n_real_rows: int) -> int:
        """
        Synthetic pool size = n_real_rows × mode.pool_multiplier.
        Always at least 1.
        """
        mc = self.get_mode_config(mode)
        return max(1, int(n_real_rows) * int(mc["pool_multiplier"]))

    def rl_steps_for_mode(self, mode: str | None) -> int:
        """RL episode step cap for the given mode."""
        return int(self.get_mode_config(mode)["rl_max_steps"])

    def patience_for_mode(self, mode: str | None) -> int:
        """Early-stopping patience for the given mode."""
        return int(self.get_mode_config(mode)["patience"])

    def curation_threshold_for_mode(self, mode: str | None) -> float:
        """Minimum verifier confidence for bandit acceptance."""
        return float(self.get_mode_config(mode)["curation_threshold"])

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
