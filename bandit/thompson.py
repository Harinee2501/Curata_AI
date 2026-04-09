# bandit/thompson.py
# -------------------------------------------------------------------
# Thompson Sampling contextual bandit.
#
# Idea: model the reward for each arm as a Beta distribution.
#   - α (alpha) = number of "successes" (positive ΔF1) + 1
#   - β (beta)  = number of "failures"  (negative ΔF1) + 1
#
# At each step, sample a reward estimate from each arm's Beta
# distribution and pick the arm with the highest sample.
# This naturally balances exploration (wide distribution = uncertain
# arms get sampled high sometimes) and exploitation (narrow
# distribution around a high mean = reliably chosen).
# -------------------------------------------------------------------

import numpy as np
from bandit.base_bandit import BaseBandit


class ThompsonBandit(BaseBandit):

    def __init__(self):
        super().__init__()
        # Beta distribution parameters for each arm
        # Initialise at (1, 1) = uniform distribution (no prior preference)
        self.alpha = np.ones(self.N_ARMS)   # successes + 1
        self.beta  = np.ones(self.N_ARMS)   # failures  + 1

    def select(self, context: np.ndarray) -> int:
        """
        Draws a sample from each arm's Beta posterior and returns the
        arm with the highest draw.  Context boosts the accept arm.
        """
        # Sample from Beta(alpha, beta) for each arm
        samples = np.random.beta(self.alpha, self.beta)   # shape (2,)

        # Context weighting: scale accept arm by sample quality
        context_weight = float(np.mean(context))
        samples[1] *= (1.0 + context_weight)

        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """
        Override base update to also update the Beta parameters.
        Treat positive ΔF1 as a success, non-positive as a failure.
        """
        super().update(arm, reward)

        if reward > 0:
            self.alpha[arm] += 1   # success
        else:
            self.beta[arm]  += 1   # failure
