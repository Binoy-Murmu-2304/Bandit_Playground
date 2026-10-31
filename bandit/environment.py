"""
Bandit environment — the "casino" the agent plays in.

Provides two environments:
  - KArmedBandit:       Standard stationary k-armed bandit
  - NonStationaryBandit: World shifts mid-game (drift simulation)
"""

import numpy as np


class KArmedBandit:
    """
    Standard k-armed bandit testbed.

    Each arm has a hidden true value drawn from N(0, 1).
    Each pull returns a noisy reward drawn from N(true_value, noise).
    The agent sees only the rewards — never the true values.
    """

    def __init__(self, k: int = 10, noise: float = 1.0, seed: int = None):
        self.k = k
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        # Hidden true values — the secret the agent must discover
        self.true_values = self.rng.normal(0, 1, size=k)
        self.pulls_per_arm = np.zeros(k)
        self.history: list[tuple[int, float]] = []

    def pull(self, action: int) -> float:
        """Pull arm `action`, return a noisy reward."""
        self.pulls_per_arm[action] += 1
        reward = float(self.rng.normal(self.true_values[action], self.noise))
        self.history.append((action, reward))
        return reward

    def optimal_action(self) -> int:
        """Index of the truly best arm (for scoring only)."""
        return int(np.argmax(self.true_values))

    def optimal_value(self) -> float:
        """True value of the best arm."""
        return float(np.max(self.true_values))

    def reset(self, seed: int = None):
        """Re-initialise with fresh true values."""
        self.rng = np.random.default_rng(seed)
        self.true_values = self.rng.normal(0, 1, size=self.k)
        self.pulls_per_arm = np.zeros(self.k)
        self.history = []


class NonStationaryBandit(KArmedBandit):
    """
    A bandit whose true values CHANGE at a specified step.

    Simulates a world where conditions drift — directly analogous
    to the data-drift / concept-drift problem in production MLOps.
    """

    def __init__(
        self,
        k: int = 10,
        noise: float = 1.0,
        change_step: int = 500,
        drift_size: float = 2.0,
        seed: int = None,
    ):
        super().__init__(k, noise, seed)
        self.change_step = change_step
        self.drift_size = drift_size
        self.current_pull = 0
        self.world_changed = False

        # Pre-generate the post-shift values so the shift is deterministic
        self.new_true_values = self.true_values + self.rng.normal(
            0, drift_size, size=k
        )

    def pull(self, action: int) -> float:
        self.current_pull += 1

        # Trigger the world shift exactly once
        if self.current_pull == self.change_step and not self.world_changed:
            self.true_values = self.new_true_values.copy()
            self.world_changed = True

        return super().pull(action)

    def reset(self, seed: int = None):
        super().reset(seed)
        self.current_pull = 0
        self.world_changed = False
        self.new_true_values = self.true_values + self.rng.normal(
            0, self.drift_size, size=self.k
        )