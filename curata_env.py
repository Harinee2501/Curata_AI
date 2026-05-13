# curata_env.py — training environment for the PPO policy (reference for inference obs).
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DataCurationEnv(gym.Env):

    metadata = {"render_modes": []}

    ACTION_GEN_SMALL = 0
    ACTION_GEN_LARGE = 1
    ACTION_FILTER    = 2
    ACTION_RETRAIN   = 3
    ACTION_EVALUATE  = 4
    ACTION_STOP      = 5

    N_ACTIONS = 6
    STATE_DIM = 29
    MAX_STEPS = 100

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.zeros(self.STATE_DIM, dtype=np.float32),
            high=np.ones(self.STATE_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self._dataset_props = {}
        self._internal = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        n_classes = rng.integers(2, 11)
        n_real_rows = rng.integers(30, 2001)
        difficulty = rng.uniform(0.0, 1.0)
        drift_rate = rng.uniform(0.0, 0.05)
        class_balance = self._sample_class_balance(rng, n_classes)

        if options and "training_progress" in options:
            progress = options["training_progress"]
            max_budget = int(50 + progress * 450)
            total_budget = int(rng.integers(30, max(31, max_budget)))
        else:
            total_budget = int(rng.integers(100, 501))

        imbalance_penalty = 1.0 - self._entropy(class_balance)
        baseline_f1 = float(
            np.clip(
                0.75
                - (difficulty * 0.35)
                - (imbalance_penalty * 0.2)
                + rng.normal(0, 0.03),
                0.10,
                0.90,
            )
        )
        target_rows = int(rng.integers(20, 201))

        self._dataset_props = {
            "n_classes": int(n_classes),
            "n_real_rows": int(n_real_rows),
            "difficulty": float(difficulty),
            "drift_rate": float(drift_rate),
            "class_balance": class_balance,
            "total_budget": int(total_budget),
            "baseline_f1": baseline_f1,
            "target_rows": target_rows,
        }

        self._internal = {
            "step": 0,
            "current_f1": baseline_f1,
            "prev_f1": baseline_f1,
            "minority_class_f1": baseline_f1 * 0.7,
            "best_f1_so_far": baseline_f1,
            "f1_vs_best": 0.0,
            "n_curated": 0,
            "n_unfiltered": 0,
            "drift_score": 0.0,
            "verifier_conf_mean": 0.5,
            "acceptance_rate": 0.5,
            "generation_success_rate": 0.8,
            "budget_used": 0,
            "n_retrains": 0,
            "steps_since_improvement": 0,
            "consecutive_no_improve": 0,
            "consecutive_gen_without_retrain": 0,
            "last_action": self.ACTION_STOP,
            "n_curated_at_last_retrain": 0,
            "steps_since_last_eval": 99,
            "_prev_retrains": 0,
            "_filtered_since_last_retrain": False,
            "_curated_before_filter": 0,
        }

        return self._get_obs(), {}

    def step(self, action: int):
        props = self._dataset_props
        state = self._internal
        rng = self.np_random

        state["step"] += 1
        state["steps_since_last_eval"] = min(state["steps_since_last_eval"] + 1, 99)
        state["last_action"] = action

        prev_f1 = state["current_f1"]
        reward = 0.0

        state["drift_score"] = float(
            np.clip(
                state["drift_score"] + props["drift_rate"] + rng.normal(0, 0.005),
                0.0,
                1.0,
            )
        )
        drift_penalty = state["drift_score"] * 0.05

        if action in [self.ACTION_GEN_SMALL, self.ACTION_GEN_LARGE]:
            state["consecutive_gen_without_retrain"] += 1
        elif action == self.ACTION_RETRAIN:
            state["consecutive_gen_without_retrain"] = 0

        if action == self.ACTION_GEN_SMALL:
            reward = self._action_generate(10, props, state, rng)
        elif action == self.ACTION_GEN_LARGE:
            reward = self._action_generate(50, props, state, rng)
        elif action == self.ACTION_FILTER:
            reward = self._action_filter(props, state, rng)
        elif action == self.ACTION_RETRAIN:
            reward = self._action_retrain(props, state, rng)
        elif action == self.ACTION_EVALUATE:
            reward = self._action_evaluate(props, state, rng)
        elif action == self.ACTION_STOP:
            reward = self._action_stop(props, state)

        reward += self._sequence_penalty(action, state)

        if (
            action == self.ACTION_RETRAIN
            and state["n_retrains"] > state["_prev_retrains"]
            and state["_filtered_since_last_retrain"]
        ):
            reward += 1.5
            state["_filtered_since_last_retrain"] = False

        if (
            action == self.ACTION_FILTER
            and state["n_curated"] > state["_curated_before_filter"]
        ):
            state["_filtered_since_last_retrain"] = True

        state["_prev_retrains"] = state["n_retrains"]
        if action == self.ACTION_FILTER:
            state["_curated_before_filter"] = state["n_curated"]

        reward -= drift_penalty
        state["prev_f1"] = prev_f1

        if state["current_f1"] > prev_f1 + 0.001:
            state["steps_since_improvement"] = 0
            state["consecutive_no_improve"] = 0
        else:
            state["steps_since_improvement"] += 1
            state["consecutive_no_improve"] += 1

        if state["current_f1"] > state["best_f1_so_far"]:
            state["best_f1_so_far"] = state["current_f1"]
        state["f1_vs_best"] = state["current_f1"] - state["best_f1_so_far"]

        stagnation = state["consecutive_no_improve"]
        if stagnation >= 6:
            reward -= min(0.05 * (stagnation - 5), 1.0)
        if stagnation >= 10:
            reward -= 0.3

        degradation = state["best_f1_so_far"] - state["current_f1"]
        if degradation > 0.05:
            reward -= degradation * 2.0

        terminated = (
            action == self.ACTION_STOP
            or state["budget_used"] >= props["total_budget"]
            or state["current_f1"] >= 0.95
        )
        truncated = state["step"] >= self.MAX_STEPS

        state["budget_used"] = min(state["budget_used"], props["total_budget"])

        obs = self._get_obs()
        info = {
            "current_f1": state["current_f1"],
            "n_curated": state["n_curated"],
            "budget_used": state["budget_used"],
            "step": state["step"],
        }

        return obs, float(reward), terminated, truncated, info

    def _action_generate(self, batch_size, props, state, rng):
        if state["budget_used"] + batch_size > props["total_budget"]:
            return -0.5
        state["budget_used"] += batch_size

        rows_total = state["n_curated"] + state["n_unfiltered"]
        target_rows = props.get("target_rows", 100)
        if rows_total > target_rows * 3 and state["n_retrains"] == 0:
            return -1.0

        base_success = 0.85 - (props["difficulty"] * 0.25)
        success_rate = float(np.clip(base_success + rng.normal(0, 0.05), 0.3, 1.0))
        state["generation_success_rate"] = success_rate

        rows_generated = int(batch_size * success_rate)
        state["n_unfiltered"] += rows_generated
        state["verifier_conf_mean"] = float(
            np.clip(0.75 - (props["difficulty"] * 0.3) + rng.normal(0, 0.05), 0.2, 1.0)
        )

        reward = rows_generated * 0.005
        if state["n_unfiltered"] > 100:
            reward -= 0.3
        elif state["n_unfiltered"] > 50:
            reward -= 0.1

        consecutive = state["consecutive_gen_without_retrain"]
        if consecutive >= 3:
            reward -= min(0.1 * (consecutive - 2), 1.0)

        return float(reward)

    def _action_filter(self, props, state, rng):
        if state["n_unfiltered"] == 0:
            return -0.2

        acceptance_rate = float(
            np.clip(state["verifier_conf_mean"] + rng.normal(0, 0.08), 0.1, 1.0)
        )
        state["acceptance_rate"] = acceptance_rate

        newly_curated = int(state["n_unfiltered"] * acceptance_rate)
        state["n_curated"] += newly_curated
        state["n_unfiltered"] = 0

        saturation = np.clip(state["n_curated"] / 500, 0, 1)

        if acceptance_rate > 0.90:
            reward = -1.5
        elif acceptance_rate > 0.85:
            reward = newly_curated * 0.015 * (1.0 - saturation * 0.5) - 0.5
        else:
            reward = newly_curated * 0.03 * (1.0 - saturation * 0.5)

        return float(reward)

    def _action_retrain(self, props, state, rng):
        new_curated = state["n_curated"] - state["n_curated_at_last_retrain"]
        if new_curated < 5:
            return -0.3

        state["n_retrains"] += 1
        state["n_curated_at_last_retrain"] = state["n_curated"]

        data_quality = state["verifier_conf_mean"]
        if state["acceptance_rate"] > 0.85:
            data_quality *= 0.7

        curated_benefit = np.log1p(new_curated) / 10.0
        difficulty_scaling = 1.0 - (props["difficulty"] * 0.5)
        balance_entropy = self._entropy(props["class_balance"])

        f1_delta = (
            curated_benefit * difficulty_scaling * data_quality * balance_entropy * 0.15
        )

        if data_quality < 0.5:
            f1_delta -= (0.5 - data_quality) * 0.3

        f1_delta += rng.normal(0, 0.02)
        f1_delta = float(np.clip(f1_delta, -0.15, 0.15))

        minority_delta = f1_delta * 1.2
        state["minority_class_f1"] = float(
            np.clip(state["minority_class_f1"] + minority_delta, 0.0, 1.0)
        )

        old_f1 = state["current_f1"]
        state["current_f1"] = float(np.clip(old_f1 + f1_delta, 0.0, 1.0))
        f1_gain = state["current_f1"] - old_f1

        reward = f1_gain * 10.0
        if f1_gain < -0.03:
            reward -= 2.0
        elif f1_gain < 0.0:
            reward -= 0.5

        reward += max(0.0, minority_delta) * 5.0

        if state["n_retrains"] > 5 and new_curated < 10:
            reward -= 0.3

        return float(reward)

    def _action_evaluate(self, props, state, rng):
        if state["budget_used"] + 2 > props["total_budget"]:
            return -0.1

        state["budget_used"] += 2
        state["steps_since_last_eval"] = 0
        state["current_f1"] = float(
            np.clip(state["current_f1"] + rng.normal(0, 0.01), 0.0, 1.0)
        )

        if state["step"] > 5 and state["step"] % 3 != 0:
            return 0.01
        return 0.05

    def _action_stop(self, props, state):
        f1_gain = state["current_f1"] - props["baseline_f1"]
        final_reward = f1_gain * 50.0

        if state["n_retrains"] == 0:
            final_reward -= 15.0

        if state["consecutive_no_improve"] >= 3 and f1_gain > 0.02:
            final_reward += 5.0

        if state["current_f1"] >= 0.85:
            final_reward += 5.0
        if state["current_f1"] >= 0.90:
            final_reward += 8.0
        if state["current_f1"] >= 0.95:
            final_reward += 12.0

        if state["n_unfiltered"] > 20:
            final_reward -= 2.0

        budget_efficiency = 1.0 - (state["budget_used"] / max(props["total_budget"], 1))
        final_reward += budget_efficiency * 0.5

        return float(final_reward)

    def _sequence_penalty(self, action, state):
        penalty = 0.0
        if action == self.ACTION_RETRAIN:
            if (state["n_curated"] - state["n_curated_at_last_retrain"]) < 5:
                penalty -= 0.5
        if action == self.ACTION_FILTER:
            if state["n_unfiltered"] < 5:
                penalty -= 0.3
        if action in [self.ACTION_GEN_SMALL, self.ACTION_GEN_LARGE]:
            if state["n_unfiltered"] > 50:
                penalty -= 0.4
        if action == self.ACTION_EVALUATE and state["steps_since_last_eval"] < 5:
            penalty -= 0.2
        return penalty

    def _get_obs(self):
        props = self._dataset_props
        state = self._internal

        last_action_onehot = np.zeros(self.N_ACTIONS, dtype=np.float32)
        last_action_onehot[state["last_action"]] = 1.0

        obs = np.array(
            [
                state["current_f1"],
                float(
                    np.clip(
                        (state["current_f1"] - state["prev_f1"]) * 0.5 + 0.5,
                        0,
                        1,
                    )
                ),
                props["baseline_f1"],
                float(np.clip(props["n_real_rows"] / 2000.0, 0, 1)),
                float(np.clip(state["n_curated"] / 500.0, 0, 1)),
                float(np.clip(state["n_unfiltered"] / 200.0, 0, 1)),
                self._entropy(props["class_balance"]),
                state["minority_class_f1"],
                state["drift_score"],
                state["verifier_conf_mean"],
                state["acceptance_rate"],
                float(1.0 - state["budget_used"] / max(props["total_budget"], 1)),
                float(state["step"] / self.MAX_STEPS),
                *last_action_onehot,
                float(np.clip(state["n_retrains"] / 20.0, 0, 1)),
                float(np.clip(state["steps_since_improvement"] / 20.0, 0, 1)),
                props["difficulty"],
                float(np.clip(props["n_classes"] / 10.0, 0, 1)),
                state["generation_success_rate"],
                float(np.clip(state["consecutive_no_improve"] / 10.0, 0, 1)),
                float(np.clip(state["consecutive_no_improve"] / 5.0, 0, 1)),
                float(np.clip(state["budget_used"] / max(props["total_budget"], 1), 0, 1)),
                float(np.clip(state["f1_vs_best"] * 0.5 + 0.5, 0, 1)),
                float(np.clip(state["consecutive_gen_without_retrain"] / 5.0, 0, 1)),
            ],
            dtype=np.float32,
        )

        obs = np.clip(obs, 0.0, 1.0)

        assert obs.shape == (self.STATE_DIM,), (
            f"State dim mismatch: expected {self.STATE_DIM}, got {obs.shape[0]}"
        )
        return obs

    @staticmethod
    def _sample_class_balance(rng, n_classes):
        concentration = rng.choice([0.3, 0.5, 1.0, 2.0, 5.0])
        alpha = np.full(n_classes, concentration)
        return rng.dirichlet(alpha).astype(np.float32)

    @staticmethod
    def _entropy(dist):
        dist = np.asarray(dist, dtype=np.float64) + 1e-9
        dist = dist / dist.sum()
        H = -np.sum(dist * np.log(dist))
        H_max = np.log(len(dist))
        return float(H / H_max) if H_max > 0 else 1.0
