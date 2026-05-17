# bandit/thompson.py
# ─────────────────────────────────────────────────────────────────────
# Thompson Sampling contextual bandit — the sole selection policy.
#
# Why Thompson Sampling over UCB for this problem:
#   - Adapts faster in low-iteration settings (20-30 rounds)
#   - Handles noisy, shifting rewards (model changes every iteration)
#   - Probabilistic exploration is more robust than UCB's fixed bonus
#
# How it works:
#   - Maintains Beta(alpha, beta) distribution per action
#   - Each call samples from these and picks the higher draw
#   - Context (quality vector) scales the accept arm's sample
#   - After each batch: alpha++ on success, beta++ on failure
#
# Reward calibration rationale
# ─────────────────────────────
# The verifier confidence is computed against a WEAK model early in the
# episode (baseline F1 ≈ 0.25 on a 3-class problem = essentially random).
# This means most rows will have ver in the 0.30–0.55 range even if they
# are genuinely useful. Thresholds must account for this:
#
#   v1 (original):  accept reward = 1.0 if ver>0.6 else 0.0  → ~100% accept
#                   reject reward = 0.0 always               → bandit never learns to reject
#   v2 (over-fix):  accept reward tiered at 0.75/0.60        → ~10–20% accept (too strict)
#                   reject reward fires at ver<0.50           → reject arm dominates early
#   v3 (this file): looser thresholds tuned for weak-model
#                   regime; target acceptance 40–60%.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
from loguru import logger


class ThompsonBandit:
    """
    Contextual Thompson Sampling bandit for accept/reject decisions.

    Actions:
        0 = reject (discard synthetic sample)
        1 = accept (add to curated pool)

    Context:
        [uncertainty, novelty, verifier_confidence] — all in [0, 1]
    """

    N_ARMS = 2

    def __init__(self, context_boost: float = 0.1):
        """
        context_boost : added to the accept-arm Beta sample in proportion to
                        verifier confidence (see ``select``). Tunable via config.
        """
        self.context_boost = float(np.clip(context_boost, 0.0, 2.0))
        self.alpha = np.ones(self.N_ARMS, dtype=np.float64)
        self.beta_p = np.ones(self.N_ARMS, dtype=np.float64)
        self.counts = np.zeros(self.N_ARMS, dtype=np.int64)
        self.rewards = np.zeros(self.N_ARMS, dtype=np.float64)

        self.total_steps = 0
        self.total_accepts = 0
        self.total_rejects = 0

        logger.info(
            f"ThompsonBandit initialised — Beta(1,1) prior, context_boost={self.context_boost}"
        )

    def reset(self):
        """
        Resets all posterior state back to Beta(1,1) priors.
        Call this when reusing the bandit across augmentation runs
        (e.g. switching between Fast / Balanced / Thorough modes).
        """
        self.alpha = np.ones(self.N_ARMS, dtype=np.float64)
        self.beta_p = np.ones(self.N_ARMS, dtype=np.float64)
        self.counts = np.zeros(self.N_ARMS, dtype=np.int64)
        self.rewards = np.zeros(self.N_ARMS, dtype=np.float64)
        self.total_steps = 0
        self.total_accepts = 0
        self.total_rejects = 0
        logger.info("ThompsonBandit reset — posteriors restored to Beta(1,1).")

    def select(self, context: np.ndarray) -> int:
        """
        Selects accept (1) or reject (0) for one synthetic sample.

        Parameters
        ----------
        context : np.ndarray([uncertainty, novelty, verifier_confidence])

        Returns
        -------
        action : int — 0 (reject) or 1 (accept)
        """
        samples = np.random.beta(self.alpha, self.beta_p)

        # Nudge accept arm by sample quality — additive so reject arm can still win.
        # Use verifier_confidence (context[2]) as the quality signal only.
        context_quality = float(np.clip(context[2], 0.0, 1.0))
        samples[1] += self.context_boost * context_quality

        action = int(np.argmax(samples))
        self.counts[action] += 1
        self.total_steps += 1

        if action == 1:
            self.total_accepts += 1
        else:
            self.total_rejects += 1

        return action

    def update(self, arm: int, reward: float):
        """
        Updates Beta posterior based on observed reward.
        Positive reward → alpha increases (success); negative/zero → beta increases (failure).
        Update magnitude scales with abs(reward) so strong signals shift the posterior more.
        """
        self.rewards[arm] += reward

        if reward > 0:
            self.alpha[arm] += abs(reward)
        elif reward < 0:
            self.beta_p[arm] += abs(reward)
        else:
            self.beta_p[arm] += 1.0  # zero reward = mild failure nudge

        logger.debug(
            f"Bandit update — arm={arm} reward={reward:+.4f} "
            f"alpha={self.alpha.tolist()} beta={self.beta_p.tolist()}"
        )

    def compute_accept_reward(self, verifier_conf: float) -> float:
        """
        Tiered reward for the ACCEPT arm based on verifier confidence.

        Thresholds are deliberately loose to account for the weak-model regime
        early in the episode (baseline F1 ≈ 0.25 on 3-class → verifier scores
        cluster around 0.30–0.55 even for genuinely useful rows).

        Targets ~40–60% overall acceptance rate (v3 thresholds):
            ver > 0.55  → 1.0   (good row relative to a weak model)
            ver > 0.40  → 0.5   (borderline — partial credit)
            ver ≤ 0.40  → 0.0   (likely noise — no credit)
        """
        v = float(np.clip(verifier_conf, 0.0, 1.0))
        if v > 0.55:
            return 1.0
        elif v > 0.40:
            return 0.5
        else:
            return 0.0

    def compute_reject_reward(self, verifier_conf: float) -> float:
        """
        Reward for the REJECT arm based on verifier confidence.

        Only reward rejection for rows that are clearly bad (very low confidence),
        to avoid the reject arm dominating when the model is weak and most rows
        legitimately score in the 0.30–0.50 range.

        Targets ~40–60% overall acceptance rate (v3 thresholds):
            ver < 0.30  → 0.8   (very low confidence — correct to reject)
            ver < 0.40  → 0.3   (minor credit for rejecting borderline row)
            ver ≥ 0.40  → 0.0   (decent row rejected — no credit; mild penalty)
        """
        v = float(np.clip(verifier_conf, 0.0, 1.0))
        if v < 0.30:
            return 0.8
        elif v < 0.40:
            return 0.3
        else:
            return 0.0

    def acceptance_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_accepts / self.total_steps

    def summary(self) -> dict:
        return {
            "total_steps": int(self.total_steps),
            "total_accepts": int(self.total_accepts),
            "total_rejects": int(self.total_rejects),
            "acceptance_rate": round(self.acceptance_rate(), 4),
            "alpha": self.alpha.tolist(),
            "beta": self.beta_p.tolist(),
            "cumulative_reward_accept": round(float(self.rewards[1]), 4),
            "cumulative_reward_reject": round(float(self.rewards[0]), 4),
        }
