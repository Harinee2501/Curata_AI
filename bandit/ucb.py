# bandit/ucb.py
# -------------------------------------------------------------------
# Upper Confidence Bound (UCB1) contextual bandit.
#
# UCB formula:
#   score(arm) = mean_reward(arm) + alpha * sqrt(ln(t) / count(arm))
#
# The second term is the "exploration bonus" — arms that have been
# tried fewer times get a higher bonus, encouraging exploration.
# As counts grow the bonus shrinks and we favour exploitation.
#
# We make it "contextual" by weighting the accept arm's base score
# with the sample's quality context vector.
# -------------------------------------------------------------------

import numpy as np
from bandit.base_bandit import BaseBandit
import config


class UCBBandit(BaseBandit):

    def __init__(self, alpha: float = config.UCB_ALPHA):
        super().__init__()
        self.alpha   = alpha       # exploration–exploitation trade-off
        self.t       = 0           # total number of selections so far

    def select(self, context: np.ndarray) -> int:
        """
        Selects accept (1) or reject (0) for one synthetic sample.

        context : np.ndarray([uncertainty, novelty, verifier_confidence])

        Context is used to scale the accept arm's expected reward:
        a high-quality context boosts the accept arm's score so we
        prefer accepting promising samples.
        """
        self.t += 1

        # Guard: explore randomly for the first few samples
        if self.t <= self.N_ARMS:
            return self.t - 1

        # UCB score for each arm
        exploration_bonus = self.alpha * np.sqrt(
            np.log(self.t) / (self.counts + 1e-9)
        )
        ucb_scores = self.rewards + exploration_bonus  # shape (2,)

        # Context weighting: the mean quality of the context vector
        # boosts the accept arm so better samples are more likely accepted
        context_weight = float(np.mean(context))
        ucb_scores[1] *= (1.0 + context_weight)

        return int(np.argmax(ucb_scores))
