# agent.py
# ─────────────────────────────────────────────────────────────────────
# RLAgent — loads the trained PPO policy and runs a real curation
# episode by mapping discrete actions to actual pipeline operations.
#
# Augmentation mode integration
# ──────────────────────────────
# The agent itself is mode-agnostic: it simply respects whatever
# `max_steps` and `total_budget` it is given. Both are derived from the
# active augmentation mode in pipeline.py before RLAgent is constructed:
#
#   fast      → max_steps=10,  budget scaled accordingly
#   balanced  → max_steps=40   (default)
#   thorough  → max_steps=100, larger budget
#
# The bandit_filter_fn passed in already embeds the mode's
# `curation_threshold`, so no extra wiring is needed here.
#
# Observation vector EXACTLY matches DataCurationEnv._get_obs() (29 floats):
#   [0]  current_f1
#   [1]  (current_f1 - prev_f1) * 0.5 + 0.5     ← delta, normalised
#   [2]  baseline_f1
#   [3]  n_real_rows / 2000
#   [4]  n_curated / 500
#   [5]  n_unfiltered / 200
#   [6]  class_balance entropy
#   [7]  minority_class_f1
#   [8]  drift_score
#   [9]  verifier_conf_mean
#   [10] acceptance_rate
#   [11] 1 - budget_used/total_budget             ← budget REMAINING
#   [12] step / MAX_STEPS
#   [13-18] one-hot last action (6 slots)
#   [19] n_retrains / 20
#   [20] steps_since_improvement / 20
#   [21] dataset_difficulty
#   [22] n_classes / 10
#   [23] generation_success_rate
#   [24] consecutive_no_improve / 10
#   [25] consecutive_no_improve / 5               ← stop_attractiveness
#   [26] budget_used / total_budget               ← budget urgency
#   [27] f1_vs_best * 0.5 + 0.5
#   [28] consecutive_gen_without_retrain / 5
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from loguru import logger

from bandit.thompson import ThompsonBandit
from config import cfg
from curata_env import DataCurationEnv

# ── ANSI colours ──────────────────────────────────────────────────────
_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "grey": "\033[90m",
}

_ACTION_COLOUR = {
    0: _C["cyan"],
    1: _C["blue"],
    2: _C["yellow"],
    3: _C["green"],
    4: _C["magenta"],
    5: _C["red"],
}

_ACTION_EMOJI = {
    0: "🔹 GEN_SMALL ",
    1: "🔷 GEN_LARGE ",
    2: "🔍 FILTER    ",
    3: "🔁 RETRAIN   ",
    4: "📊 EVALUATE  ",
    5: "🛑 STOP      ",
}

# ── Override thresholds ───────────────────────────────────────────────
_FORCE_FILTER_UNFILTERED_MIN = 10
_FORCE_RETRAIN_CURATED_MIN = 20
_MAX_CONSECUTIVE_EVALUATE = 3


def _class_balance_entropy_from_labels(y_train: np.ndarray, num_classes: int) -> float:
    y = np.asarray(y_train, dtype=int).ravel()
    cnt = np.bincount(y, minlength=max(num_classes, 1))
    tot = float(cnt.sum())
    if tot <= 0:
        return 1.0
    p = (cnt.astype(np.float64) / tot).astype(np.float32)
    return DataCurationEnv._entropy(p)


# ─────────────────────────────────────────────────────────────────────
# EPISODE STATE
# ─────────────────────────────────────────────────────────────────────


class _EpisodeState:
    def __init__(
        self,
        baseline_f1: float,
        n_real_rows: int,
        num_classes: int,
        total_budget: int,
        target_size: int,
        class_balance_entropy: float,
        dataset_difficulty: float,
    ):
        self.baseline_f1 = float(baseline_f1)
        self.n_real_rows = int(n_real_rows)
        self.num_classes = int(num_classes)
        self.total_budget = int(total_budget)
        self.target_size = max(int(target_size), 1)
        self.class_balance_entropy = float(np.clip(class_balance_entropy, 0.0, 1.0))
        self.dataset_difficulty = float(np.clip(dataset_difficulty, 0.0, 1.0))

        self.current_f1 = float(baseline_f1)
        self.prev_f1 = float(baseline_f1)
        self.best_f1_so_far = float(baseline_f1)
        self.minority_class_f1 = float(baseline_f1) * 0.7

        self.n_curated = 0
        self.n_unfiltered = 0

        self.drift_score = 0.0
        self.verifier_conf_mean = 0.5
        self.acceptance_rate = 0.5
        self.generation_success_rate = 0.8

        self.budget_used = 0
        self.n_retrains = 0

        self.steps_since_improvement = 0
        self.consecutive_no_improve = 0
        self.consecutive_gen_without_retrain = 0

        self.last_action = DataCurationEnv.ACTION_STOP

        self.episode_step = 0

        self.consecutive_evaluate = 0
        self.curated_since_last_retrain = 0

    @property
    def f1_vs_best(self) -> float:
        return self.current_f1 - self.best_f1_so_far

    def observe_f1(self, new_f1: float) -> None:
        nf = float(new_f1)
        self.prev_f1 = self.current_f1
        self.current_f1 = nf
        if nf > self.best_f1_so_far:
            self.best_f1_so_far = nf

    def to_obs(self) -> np.ndarray:
        last_action_onehot = np.zeros(DataCurationEnv.N_ACTIONS, dtype=np.float32)
        la = int(np.clip(self.last_action, 0, DataCurationEnv.N_ACTIONS - 1))
        last_action_onehot[la] = 1.0

        stop_attractiveness = float(np.clip(self.consecutive_no_improve / 5.0, 0, 1))
        budget_urgency = float(
            np.clip(self.budget_used / max(self.total_budget, 1), 0, 1)
        )

        obs = np.array(
            [
                self.current_f1,
                float(
                    np.clip(
                        (self.current_f1 - self.prev_f1) * 0.5 + 0.5,
                        0,
                        1,
                    )
                ),
                self.baseline_f1,
                float(np.clip(self.n_real_rows / 2000.0, 0, 1)),
                float(np.clip(self.n_curated / 500.0, 0, 1)),
                float(np.clip(self.n_unfiltered / 200.0, 0, 1)),
                self.class_balance_entropy,
                self.minority_class_f1,
                self.drift_score,
                self.verifier_conf_mean,
                self.acceptance_rate,
                float(1.0 - self.budget_used / max(self.total_budget, 1)),
                float(self.episode_step / DataCurationEnv.MAX_STEPS),
                *last_action_onehot,
                float(np.clip(self.n_retrains / 20.0, 0, 1)),
                float(np.clip(self.steps_since_improvement / 20.0, 0, 1)),
                self.dataset_difficulty,
                float(np.clip(self.num_classes / 10.0, 0, 1)),
                self.generation_success_rate,
                float(np.clip(self.consecutive_no_improve / 10.0, 0, 1)),
                stop_attractiveness,
                budget_urgency,
                float(np.clip(self.f1_vs_best * 0.5 + 0.5, 0, 1)),
                float(np.clip(self.consecutive_gen_without_retrain / 5.0, 0, 1)),
            ],
            dtype=np.float32,
        )

        return np.clip(obs, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# VecNormalize loader
# ─────────────────────────────────────────────────────────────────────


def _load_vec_normalize_for_inference(vecnormalize_path: str):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    path = Path(vecnormalize_path)
    with path.open("rb") as f:
        vn = pickle.load(f)

    if not isinstance(vn, VecNormalize):
        raise TypeError(
            f"{path} is not a VecNormalize pickle (got {type(vn).__name__})"
        )

    obs_sp = vn.observation_space
    if not isinstance(obs_sp, spaces.Box):
        raise ValueError(
            f"VecNormalize observation_space must be Box, got {type(obs_sp)}"
        )

    shape = obs_sp.shape
    act_sp = vn.action_space
    n_actions = int(act_sp.n) if isinstance(act_sp, spaces.Discrete) else 6

    def _make_env():
        class _StubEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )
                self.action_space = spaces.Discrete(n_actions)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(shape, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(shape, dtype=np.float32), 0.0, False, False, {}

        return _StubEnv()

    venv = DummyVecEnv([_make_env])
    vn.set_venv(venv)
    vn.training = False
    vn.norm_reward = False
    return vn


# ─────────────────────────────────────────────────────────────────────
# LOGGING HELPERS
# ─────────────────────────────────────────────────────────────────────


def _log_decision(step: int, action: int, state: _EpisodeState, extra: str = ""):
    colour = _ACTION_COLOUR.get(action, "")
    emoji = _ACTION_EMOJI.get(action, f"   ACTION {action} ")
    reset = _C["reset"]
    grey = _C["grey"]
    bold = _C["bold"]
    budget_pct = 100 * (1 - state.budget_used / max(state.total_budget, 1))
    line = (
        f"{bold}[step {step:>3}]{reset} "
        f"{colour}{emoji}{reset}  "
        f"{grey}f1={state.current_f1:.4f}  "
        f"curated={state.n_curated:<4}  "
        f"unfiltered={state.n_unfiltered:<4}  "
        f"budget={state.budget_used}/{state.total_budget} ({budget_pct:.0f}% left)"
        f"{reset}"
    )
    if extra:
        line += f"  {_C['green']}{extra}{reset}"
    logger.info(line)


# ─────────────────────────────────────────────────────────────────────
# ACTION OVERRIDE LOGIC
# ─────────────────────────────────────────────────────────────────────


def _override_action(
    policy_action: int,
    state: _EpisodeState,
    unfiltered_len: int,
    curated_len: int,
) -> tuple[int, str]:
    needs_more_curated = curated_len < state.target_size
    budget_remaining = state.total_budget - state.budget_used

    if (
        unfiltered_len >= _FORCE_FILTER_UNFILTERED_MIN
        and policy_action != DataCurationEnv.ACTION_FILTER
    ):
        return (
            DataCurationEnv.ACTION_FILTER,
            f"[OVERRIDE→FILTER] {unfiltered_len} rows waiting in pool",
        )

    if (
        state.curated_since_last_retrain >= _FORCE_RETRAIN_CURATED_MIN
        and curated_len > 0
        and policy_action
        not in (
            DataCurationEnv.ACTION_RETRAIN,
            DataCurationEnv.ACTION_STOP,
        )
    ):
        return (
            DataCurationEnv.ACTION_RETRAIN,
            f"[OVERRIDE→RETRAIN] {state.curated_since_last_retrain} new curated rows",
        )

    if (
        policy_action == DataCurationEnv.ACTION_EVALUATE
        and state.consecutive_evaluate >= _MAX_CONSECUTIVE_EVALUATE
        and needs_more_curated
    ):
        if unfiltered_len > 0:
            return (
                DataCurationEnv.ACTION_FILTER,
                f"[OVERRIDE→FILTER] broke EVALUATE loop ({state.consecutive_evaluate} in a row)",
            )
        batch = 50 if budget_remaining >= 50 else 10
        action = (
            DataCurationEnv.ACTION_GEN_LARGE
            if batch == 50
            else DataCurationEnv.ACTION_GEN_SMALL
        )
        if budget_remaining >= batch:
            return (action, "[OVERRIDE→GEN] broke EVALUATE loop, pool empty")

    if (
        policy_action == DataCurationEnv.ACTION_STOP
        and needs_more_curated
        and budget_remaining >= 10
    ):
        if unfiltered_len >= _FORCE_FILTER_UNFILTERED_MIN:
            return (
                DataCurationEnv.ACTION_FILTER,
                "[OVERRIDE→FILTER] blocked premature STOP, pool has rows",
            )
        batch = 50 if budget_remaining >= 50 else 10
        action = (
            DataCurationEnv.ACTION_GEN_LARGE
            if batch == 50
            else DataCurationEnv.ACTION_GEN_SMALL
        )
        return (action, "[OVERRIDE→GEN] blocked premature STOP, need more rows")

    return policy_action, ""


# ─────────────────────────────────────────────────────────────────────
# RLAGENT
# ─────────────────────────────────────────────────────────────────────


class RLAgent:
    """
    Wraps a trained SB3 PPO model for inference against a real pipeline.

    Parameters
    ----------
    policy_path : str
        Path to best_model.zip (saved with model.save()).
    max_steps : int
        Hard episode cap. Set by the active augmentation mode in pipeline.py:
          fast=10, balanced=40, thorough=100.
    vecnormalize_path : str | None
        Path to vecnormalize_final.pkl from the same training run.
    """

    ACTION_GEN_SMALL = DataCurationEnv.ACTION_GEN_SMALL
    ACTION_GEN_LARGE = DataCurationEnv.ACTION_GEN_LARGE
    ACTION_FILTER = DataCurationEnv.ACTION_FILTER
    ACTION_RETRAIN = DataCurationEnv.ACTION_RETRAIN
    ACTION_EVALUATE = DataCurationEnv.ACTION_EVALUATE
    ACTION_STOP = DataCurationEnv.ACTION_STOP
    STATE_DIM = DataCurationEnv.STATE_DIM

    def __init__(
        self,
        policy_path: str,
        max_steps: int = 40,
        vecnormalize_path: str | None = None,
    ):
        from stable_baselines3 import PPO

        logger.info(f"[RLAgent] Loading policy from {policy_path}")
        self.model = PPO.load(policy_path, device="cpu")
        self.max_steps = max_steps

        self._vec_normalize = None
        policy_dim = int(np.prod(self.model.observation_space.shape))

        if vecnormalize_path:
            p = Path(vecnormalize_path)
            if p.is_file():
                try:
                    self._vec_normalize = _load_vec_normalize_for_inference(
                        str(p.resolve())
                    )
                    vn_dim = int(np.prod(self._vec_normalize.observation_space.shape))
                    if vn_dim != policy_dim:
                        logger.warning(
                            f"[RLAgent] VecNormalize obs dim {vn_dim} != "
                            f"policy obs dim {policy_dim} — mismatch!"
                        )
                    logger.info(f"[RLAgent] VecNormalize loaded ✓ (dim={vn_dim})")
                except Exception:
                    logger.exception("[RLAgent] Failed to load VecNormalize")
                    raise
            else:
                logger.warning(
                    f"[RLAgent] vecnormalize_path set but file not found: {p}\n"
                    "  Observations will NOT be normalised — policy will behave incorrectly."
                )

        if policy_dim != DataCurationEnv.STATE_DIM:
            raise ValueError(
                f"[RLAgent] Policy expects obs dim {policy_dim} but "
                f"DataCurationEnv.STATE_DIM={DataCurationEnv.STATE_DIM}. "
                "The model and env are mismatched."
            )

        logger.info(
            f"[RLAgent] Policy ready ✓  (obs_dim={policy_dim}, max_steps={max_steps})"
        )

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self._vec_normalize is None:
            return obs
        batch = obs.reshape(1, -1)
        normalised = self._vec_normalize.normalize_obs(batch)
        return normalised[0]

    # ─────────────────────────────────────────────────────────────────
    # MAIN EPISODE RUNNER
    # ─────────────────────────────────────────────────────────────────

    def run_episode(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        df,
        label_col: str,
        label_names: list,
        content_cols: list,
        num_classes: int,
        build_model_fn,
        train_fn,
        evaluate_f1_fn,
        get_probs_fn,
        generate_fn,
        bandit_filter_fn,
        target_size: int = 100,
        total_budget: int = 300,
        rows_to_x_batch=None,
    ) -> dict:

        model = build_model_fn(num_classes)
        train_fn(model, (X_train, y_train))
        baseline_f1 = evaluate_f1_fn(model, (X_val, y_val))

        if rows_to_x_batch is None:

            def rows_to_x_batch(rows):
                return np.array(
                    [[float(r[c]) for c in content_cols] for r in rows],
                    dtype=np.float64,
                )

        cb_entropy = _class_balance_entropy_from_labels(y_train, num_classes)
        difficulty = float(np.clip(1.0 - baseline_f1, 0.0, 1.0))

        state = _EpisodeState(
            baseline_f1=baseline_f1,
            n_real_rows=len(X_train),
            num_classes=num_classes,
            total_budget=total_budget,
            target_size=target_size,
            class_balance_entropy=cb_entropy,
            dataset_difficulty=difficulty,
        )

        curated_rows = []
        curated_X = []
        curated_y = []
        unfiltered = []
        bandit_info = {}
        step_log = []
        episode_bandit = ThompsonBandit(context_boost=cfg.THOMPSON_CONTEXT_BOOST)

        probe = state.to_obs()
        expected = int(np.prod(self.model.observation_space.shape))
        if probe.shape[0] != expected:
            raise ValueError(
                f"Obs length {probe.shape[0]} != policy expectation {expected}. "
                "Check _EpisodeState.to_obs() vs DataCurationEnv._get_obs()."
            )

        logger.info("")
        logger.info("━" * 70)
        logger.info(
            f"  {_C['bold']}Curata AI — RL Episode Starting{_C['reset']}\n"
            f"  baseline_f1={baseline_f1:.4f}  budget={total_budget}  "
            f"target={target_size} rows  max_steps={self.max_steps}"
        )
        logger.info("━" * 70)

        for step in range(self.max_steps):
            prev_f1_this_step = state.current_f1
            obs = state.to_obs()
            policy_action = int(
                self.model.predict(self._normalise_obs(obs), deterministic=True)[0]
            )

            action, override_reason = _override_action(
                policy_action=policy_action,
                state=state,
                unfiltered_len=len(unfiltered),
                curated_len=len(curated_rows),
            )
            if override_reason:
                logger.info(f"  {_C['yellow']}{override_reason}{_C['reset']}")

            if action in (self.ACTION_GEN_SMALL, self.ACTION_GEN_LARGE):
                state.consecutive_gen_without_retrain += 1
            elif action == self.ACTION_RETRAIN:
                state.consecutive_gen_without_retrain = 0

            if action == self.ACTION_EVALUATE:
                state.consecutive_evaluate += 1
            else:
                state.consecutive_evaluate = 0

            _log_decision(step, action, state, "executing...")

            log_entry = {
                "step": step,
                "action": action,
                "action_name": _action_name(action),
                "policy_action": policy_action,
                "overridden": action != policy_action,
            }
            t0 = time.time()
            extra_note = ""

            # ── Dispatch ──────────────────────────────────────────────
            if action == self.ACTION_GEN_SMALL:
                new_rows = self._do_generate(
                    df,
                    label_col,
                    label_names,
                    content_cols,
                    batch_size=10,
                    state=state,
                    generate_fn=generate_fn,
                )
                unfiltered.extend(new_rows)
                extra_note = f"+{len(new_rows)} rows → unfiltered pool"

            elif action == self.ACTION_GEN_LARGE:
                new_rows = self._do_generate(
                    df,
                    label_col,
                    label_names,
                    content_cols,
                    batch_size=50,
                    state=state,
                    generate_fn=generate_fn,
                )
                unfiltered.extend(new_rows)
                extra_note = f"+{len(new_rows)} rows → unfiltered pool"

            elif action == self.ACTION_FILTER:
                if unfiltered:
                    n_pool = len(unfiltered)
                    accepted, acc_rate, vc_mean, b_info = bandit_filter_fn(
                        rows=unfiltered,
                        model=model,
                        content_cols=content_cols,
                        label_col=label_col,
                        label_names=label_names,
                        get_probs_fn=get_probs_fn,
                        bandit=episode_bandit,
                    )
                    bandit_info = b_info
                    curated_rows.extend(accepted)
                    state.curated_since_last_retrain += len(accepted)

                    if accepted:
                        xb = rows_to_x_batch(accepted)
                        for j in range(len(accepted)):
                            curated_X.append(
                                np.asarray(xb[j], dtype=np.float64).tolist()
                            )
                            curated_y.append(
                                {l: i for i, l in enumerate(label_names)}.get(
                                    str(accepted[j][label_col]), 0
                                )
                            )

                    state.n_curated = len(curated_rows)
                    state.n_unfiltered = 0
                    state.acceptance_rate = acc_rate
                    state.verifier_conf_mean = vc_mean
                    unfiltered = []
                    log_entry["accepted"] = len(accepted)
                    log_entry["acc_rate"] = round(acc_rate, 3)
                    extra_note = (
                        f"accepted {len(accepted)}/{n_pool} (rate={acc_rate:.0%})"
                    )
                else:
                    extra_note = "⚠ pool empty — nothing to filter"

            elif action == self.ACTION_RETRAIN:
                if curated_X:
                    Xc = np.array(curated_X)
                    yc = np.array(curated_y)
                    model = build_model_fn(num_classes)
                    train_fn(
                        model,
                        (np.vstack([X_train, Xc]), np.concatenate([y_train, yc])),
                    )
                    new_f1 = evaluate_f1_fn(model, (X_val, y_val))
                    state.observe_f1(new_f1)
                    state.n_retrains += 1
                    state.curated_since_last_retrain = 0
                    state.minority_class_f1 = float(
                        np.clip(
                            state.minority_class_f1
                            + (new_f1 - prev_f1_this_step) * 1.2,
                            0.0,
                            1.0,
                        )
                    )
                    log_entry["val_f1"] = round(new_f1, 4)
                    log_entry["delta_f1"] = round(new_f1 - prev_f1_this_step, 4)
                    sign = "▲" if new_f1 >= prev_f1_this_step else "▼"
                    extra_note = (
                        f"val_f1={new_f1:.4f}  "
                        f"{sign}Δ={new_f1 - prev_f1_this_step:+.4f}"
                    )
                else:
                    extra_note = "⚠ no curated rows yet — skipped"

            elif action == self.ACTION_EVALUATE:
                f1 = evaluate_f1_fn(model, (X_val, y_val))
                state.observe_f1(f1)
                log_entry["val_f1"] = round(f1, 4)
                extra_note = f"val_f1={f1:.4f}"

            elif action == self.ACTION_STOP:
                _log_decision(step, action, state, "agent chose STOP")
                step_log.append(log_entry)
                break

            # ── Per-step state updates ────────────────────────────────
            if action not in (self.ACTION_RETRAIN, self.ACTION_EVALUATE):
                state.prev_f1 = prev_f1_this_step

            state.last_action = action
            state.budget_used += _budget_cost(action)
            state.drift_score = float(np.clip(state.drift_score + 0.0003, 0.0, 1.0))

            if state.current_f1 > prev_f1_this_step + 0.001:
                state.steps_since_improvement = 0
                state.consecutive_no_improve = 0
            else:
                state.steps_since_improvement += 1
                state.consecutive_no_improve += 1

            state.episode_step += 1
            log_entry["elapsed_s"] = round(time.time() - t0, 2)
            step_log.append(log_entry)

            _log_decision(step, action, state, extra_note)

            if state.budget_used >= total_budget:
                logger.info(
                    f"  {_C['red']}✖ Budget exhausted "
                    f"({state.budget_used}/{total_budget}){_C['reset']}"
                )
                break

            if len(curated_rows) >= target_size:
                logger.info(
                    f"  ✔ Target reached: {len(curated_rows)}/{target_size} "
                    "curated rows — agent continuing."
                )

        # ── Final retrain + test eval ─────────────────────────────────
        if curated_X:
            Xc = np.array(curated_X)
            yc = np.array(curated_y)
            final_m = build_model_fn(num_classes)
            train_fn(
                final_m,
                (np.vstack([X_train, Xc]), np.concatenate([y_train, yc])),
            )
            test_f1 = evaluate_f1_fn(final_m, (X_test, y_test))
        else:
            test_f1 = evaluate_f1_fn(model, (X_test, y_test))

        f1_gain = test_f1 - baseline_f1
        sign = "▲" if f1_gain >= 0 else "▼"

        logger.info("━" * 70)
        logger.info(
            f"  {_C['bold']}Episode Complete{_C['reset']}\n"
            f"  baseline_f1  = {baseline_f1:.4f}\n"
            f"  test_f1      = {test_f1:.4f}   {sign} gain = {f1_gain:+.4f}\n"
            f"  curated rows = {len(curated_rows)}\n"
            f"  retrains     = {state.n_retrains}\n"
            f"  steps        = {state.episode_step}\n"
            f"  budget used  = {state.budget_used} / {total_budget}"
        )
        logger.info("━" * 70)

        return {
            "curated": curated_rows,
            "final_f1": float(test_f1),
            "baseline_f1": float(baseline_f1),
            "f1_gain": float(f1_gain),
            "budget_used": int(state.budget_used),
            "total_budget": int(total_budget),
            "n_retrains": int(state.n_retrains),
            "step_log": step_log,
            "bandit_info": bandit_info,
        }

    # ─────────────────────────────────────────────────────────────────
    # GENERATE HELPER
    # ─────────────────────────────────────────────────────────────────

    def _do_generate(
        self,
        df,
        label_col,
        label_names,
        content_cols,
        batch_size,
        state,
        generate_fn,
    ) -> list:

        if state.budget_used + batch_size > state.total_budget:
            logger.info(
                f"  {_C['red']}✖ Budget too low to generate "
                f"{batch_size} rows — skipped{_C['reset']}"
            )
            return []

        logger.info(
            f"  {_C['cyan']}[GEN] Requesting {batch_size} synthetic rows...{_C['reset']}"
        )
        t0 = time.time()
        rows = generate_fn(
            df=df,
            label_col=label_col,
            label_names=label_names,
            content_cols=content_cols,
            target_size=batch_size,
            force_regen=True,
        )
        elapsed = time.time() - t0
        logger.info(
            f"  {_C['green']}[GEN] {len(rows)} rows received "
            f"in {elapsed:.2f}s{_C['reset']}"
        )
        if rows:
            logger.info(
                f"  {_C['grey']}[GEN] sample: {str(rows[0])[:200]}{_C['reset']}"
            )

        state.n_unfiltered += len(rows)
        state.generation_success_rate = len(rows) / max(batch_size, 1)
        return rows


# ─────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────


def _action_name(action: int) -> str:
    return {
        0: "gen_small",
        1: "gen_large",
        2: "filter",
        3: "retrain",
        4: "evaluate",
        5: "stop",
    }.get(action, "unknown")


def _budget_cost(action: int) -> int:
    return {0: 10, 1: 50, 2: 0, 3: 0, 4: 2, 5: 0}.get(action, 0)
