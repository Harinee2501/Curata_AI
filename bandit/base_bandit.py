# bandit/base_bandit.py
# -------------------------------------------------------------------
# Abstract base class for contextual bandit agents.
# Both UCB and Thompson Sampling inherit from this.
#
# The bandit models two "arms" (actions):
#   arm 0 → REJECT  (discard the synthetic sample)
#   arm 1 → ACCEPT  (add it to the training pool)
#
# Context vector fed per sample:
#   [uncertainty, novelty, verifier_confidence]  — all in [0, 1]
# -------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np


class BaseBandit(ABC):
    """
    Minimal interface every bandit must implement.
    """

    N_ARMS = 2   # 0 = reject, 1 = accept

    def __init__(self):
        # Running statistics updated after every reward signal
        self.counts  = np.zeros(self.N_ARMS)   # times each arm was pulled
        self.rewards = np.zeros(self.N_ARMS)   # cumulative reward per arm

    @abstractmethod
    def select(self, context: np.ndarray) -> int:
        """
        Given a 3-d context vector, return 0 (reject) or 1 (accept).
        """
        ...

    def update(self, arm: int, reward: float):
        """
        Updates the running statistics after observing a reward.
        Called once per bandit iteration (not per sample) with the
        ΔF1 reward.
        """
        self.counts[arm]  += 1
        # Incremental mean update: new_mean = old_mean + (reward - old_mean) / n
        n = self.counts[arm]
        self.rewards[arm] += (reward - self.rewards[arm]) / n

    @property
    def mean_rewards(self) -> np.ndarray:
        """Current estimated mean reward per arm."""
        return self.rewards.copy()
