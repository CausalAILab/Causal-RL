import numpy as np

class CtfUCB:
    def __init__(self, arms, alpha=1.0):
        self.arms = arms
        self.alpha = alpha
        self.counts = {}  # (context, arm) -> count
        self.values = {}  # (context, arm) -> average reward
        self.total = 0    # total rounds

    def select_arm(self, intuition):
        self.total += 1
        best_arm = None
        best_ucb = -float('inf')

        for arm in self.arms:
            key = (intuition, arm)
            n = self.counts.get(key, 0)
            value = self.values.get(key, 0.0)

            if n == 0:
                return arm  # explore untried arm
            ucb = value + self.alpha * np.sqrt(np.log(self.total) / n)

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm

    def update(self, intuition, arm, reward):
        key = (intuition, arm)
        n = self.counts.get(key, 0)
        value = self.values.get(key, 0.0)

        # Incremental mean update
        self.counts[key] = n + 1
        self.values[key] = value + (reward - value) / (n + 1)


class UCB:
    def __init__(self, arms, alpha=1.0):
        """
        arms: list of arm names (e.g., ['M1', 'M2'])
        alpha: exploration parameter
        """
        self.arms = arms
        self.alpha = alpha
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total = 0

    def select_arm(self):
        self.total += 1
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm  # explore untried arm

        ucb_scores = {
            arm: self.values[arm] + self.alpha * np.sqrt(np.log(self.total) / self.counts[arm])
            for arm in self.arms
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm, reward):
        n = self.counts[arm]
        value = self.values[arm]
        self.counts[arm] = n + 1
        self.values[arm] = value + (reward - value) / (n + 1)