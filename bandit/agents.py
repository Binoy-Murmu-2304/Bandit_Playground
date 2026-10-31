"""
Bandit agents — five action-selection strategies.

All agents share the same learning rule (update method) and differ
only in how they select actions — exactly as in Sutton & Barto Ch.2.

Agents:
  - GreedyAgent               : pure exploitation
  - EpsilonGreedyAgent        : ε probability of random exploration
  - OptimisticAgent           : inflated init drives curiosity
  - UCBAgent                  : uncertainty-driven exploration
  - ThompsonSamplingAgent     : Bayesian posterior sampling
"""

import numpy as np


class BaseAgent:
    """
    Base class shared by all agents.

    Handles Q/N bookkeeping and the incremental update rule.
    Subclasses only need to implement select_action().
    """

    def __init__(self, k: int = 10, alpha: float = None, seed: int = None):
        self.k = k
        self.alpha = alpha  # None → sample average (1/N); float → constant step
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros(k)  # Q_t(a): estimated value of each arm
        self.N = np.zeros(k)  # N_t(a): pull count per arm
        self.total_pulls = 0

    def select_action(self) -> int:
        raise NotImplementedError("Subclasses must implement select_action()")

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.total_pulls += 1

        error = reward - self.Q[action]

        if self.alpha is not None:
            # Constant step-size (adapts to non-stationary worlds)
            self.Q[action] += self.alpha * error
        else:
            # Sample average (1/N step-size)
            self.Q[action] += (1.0 / self.N[action]) * error

    def get_name(self) -> str:
        return self.__class__.__name__


# ── Greedy ──────────────────────────────────────────────────────

class GreedyAgent(BaseAgent):
    """Always picks the highest-estimate arm. Never explores."""

    def select_action(self) -> int:
        return int(np.argmax(self.Q))

    def get_name(self) -> str:
        return "Greedy (ε=0)"


# ── ε-Greedy ────────────────────────────────────────────────────

class EpsilonGreedyAgent(BaseAgent):
    """
    Exploits with probability (1-ε), explores uniformly with probability ε.
    Classic baseline for the exploration-exploitation tradeoff.
    """

    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.1,
        alpha: float = None,
        seed: int = None,
    ):
        super().__init__(k, alpha, seed)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.Q))

    def get_name(self) -> str:
        return f"ε-greedy (ε={self.epsilon})"


# ── Optimistic Initial Values ───────────────────────────────────

class OptimisticAgent(BaseAgent):
    """
    Starts all Q values at an inflated level.
    Every untried arm looks amazing → natural curiosity-driven exploration.
    No randomness required.
    """

    def __init__(
        self,
        k: int = 10,
        initial_value: float = 5.0,
        alpha: float = None,
        seed: int = None,
    ):
        super().__init__(k, alpha, seed)
        self.Q = np.full(k, initial_value)
        self.initial_value = initial_value

    def select_action(self) -> int:
        return int(np.argmax(self.Q))

    def get_name(self) -> str:
        return f"Optimistic (init={self.initial_value:.1f})"


# ── UCB (Upper Confidence Bound) ────────────────────────────────

class UCBAgent(BaseAgent):
    """
    Adds an uncertainty bonus to each arm's estimate:
        score(a) = Q(a) + c · √( ln(t) / N(a) )
    Under-tried arms get a big bonus → intelligent exploration.
    """

    def __init__(
        self,
        k: int = 10,
        c: float = 2.0,
        alpha: float = None,
        seed: int = None,
    ):
        super().__init__(k, alpha, seed)
        self.c = c

    def select_action(self) -> int:
        # First k pulls: try each arm once (avoids division by zero)
        if self.total_pulls < self.k:
            return int(self.total_pulls)

        # UCB score
        bonus = self.c * np.sqrt(
            np.log(self.total_pulls + 1) / (self.N + 1e-8)
        )
        ucb_values = self.Q + bonus
        return int(np.argmax(ucb_values))

    def get_name(self) -> str:
        return f"UCB (c={self.c})"


# ── Thompson Sampling ───────────────────────────────────────────

class ThompsonSamplingAgent(BaseAgent):
    """
    Bayesian approach with a Gaussian reward model.

    For each arm, maintains a posterior N(Q(a), σ²/N(a)).
    On each step: sample from every arm's posterior, pick the max.
    Arms with high uncertainty naturally get explored more.
    """

    def __init__(
        self,
        k: int = 10,
        noise_std: float = 1.0,
        alpha: float = None,
        seed: int = None,
    ):
        super().__init__(k, alpha, seed)
        self.noise_std = noise_std

    def select_action(self) -> int:
        samples = np.zeros(self.k)
        for a in range(self.k):
            if self.N[a] == 0:
                # No data yet → sample from prior N(0, 1)
                samples[a] = self.rng.normal(0, 1)
            else:
                uncertainty = self.noise_std / np.sqrt(self.N[a])
                samples[a] = self.rng.normal(self.Q[a], uncertainty)
        return int(np.argmax(samples))

    def get_name(self) -> str:
        return "Thompson Sampling"