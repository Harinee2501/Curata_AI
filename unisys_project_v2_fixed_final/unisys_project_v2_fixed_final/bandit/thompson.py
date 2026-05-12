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

    def __init__(self):
        self.alpha  = np.ones(self.N_ARMS, dtype=np.float64)
        self.beta_p = np.ones(self.N_ARMS, dtype=np.float64)
        self.counts  = np.zeros(self.N_ARMS, dtype=np.int64)
        self.rewards = np.zeros(self.N_ARMS, dtype=np.float64)

        self.total_steps   = 0
        self.total_accepts = 0
        self.total_rejects = 0

        logger.info("ThompsonBandit initialised — Beta(1,1) prior on both arms")

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
        # Fix: use verifier_confidence (context[2]) as the quality signal, NOT the mean
        # of all 3 features. High uncertainty (context[0]) is ambiguous — it can mean
        # informative OR noisy. Verifier confidence is the only feature where
        # higher unambiguously means "this sample is trustworthy".
        context_quality = float(np.clip(context[2], 0.0, 1.0))   # index 2 = verifier_confidence
        samples[1] += 0.3 * context_quality

        action = int(np.argmax(samples))
        self.counts[action] += 1
        self.total_steps    += 1

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
            self.alpha[arm]  += abs(reward)            # strong positive → bigger alpha shift
        elif reward < 0:
            self.beta_p[arm] += abs(reward)            # negative reward penalises the arm
        else:
            self.beta_p[arm] += 1.0                    # zero reward = mild failure nudge

        logger.debug(
            f"Bandit update — arm={arm} reward={reward:+.4f} "
            f"alpha={self.alpha.tolist()} beta={self.beta_p.tolist()}"
        )

    def acceptance_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_accepts / self.total_steps

    def summary(self) -> dict:
        return {
            "total_steps":              int(self.total_steps),
            "total_accepts":            int(self.total_accepts),
            "total_rejects":            int(self.total_rejects),
            "acceptance_rate":          round(self.acceptance_rate(), 4),
            "alpha":                    self.alpha.tolist(),
            "beta":                     self.beta_p.tolist(),
            "cumulative_reward_accept": round(float(self.rewards[1]), 4),
            "cumulative_reward_reject": round(float(self.rewards[0]), 4),
        }
