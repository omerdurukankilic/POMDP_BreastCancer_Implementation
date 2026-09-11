"""Core discrete POMDP model and belief update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class POMDP:
    """A discrete, finite-horizon Partially Observable Markov Decision Process.

    Attributes:
        states: names of the hidden states.
        actions: names of the available actions.
        observations: names of the possible observations.
        transition: array of shape (n_actions, n_states, n_states), where
            transition[a, s, s'] is P(s' | s, a).
        observation: array of shape (n_actions, n_states, n_observations),
            where observation[a, s', o] is P(o | s', a).
        reward: array of shape (n_actions, n_states) giving the immediate
            reward for taking an action in a state.
        discount: per-step discount factor in (0, 1].
    """

    states: list[str]
    actions: list[str]
    observations: list[str]
    transition: np.ndarray
    observation: np.ndarray
    reward: np.ndarray
    discount: float = 0.95

    def __post_init__(self) -> None:
        n_s, n_a, n_o = len(self.states), len(self.actions), len(self.observations)
        if self.transition.shape != (n_a, n_s, n_s):
            raise ValueError(f"transition must have shape ({n_a}, {n_s}, {n_s})")
        if self.observation.shape != (n_a, n_s, n_o):
            raise ValueError(f"observation must have shape ({n_a}, {n_s}, {n_o})")
        if self.reward.shape != (n_a, n_s):
            raise ValueError(f"reward must have shape ({n_a}, {n_s})")

    def initial_belief(self) -> np.ndarray:
        """Return a uniform belief over states."""
        n_s = len(self.states)
        return np.full(n_s, 1.0 / n_s)

    def update_belief(self, belief: np.ndarray, action_idx: int, obs_idx: int) -> np.ndarray:
        """Bayesian belief update given an action taken and an observation received."""
        predicted = belief @ self.transition[action_idx]
        likelihood = self.observation[action_idx, :, obs_idx]
        unnormalized = predicted * likelihood
        total = unnormalized.sum()
        if total <= 0:
            raise ValueError("observation has zero probability under the predicted belief")
        return unnormalized / total

    def expected_reward(self, belief: np.ndarray, action_idx: int) -> float:
        """Expected immediate reward of an action under a belief state."""
        return float(belief @ self.reward[action_idx])
