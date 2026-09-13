"""Exact finite-horizon POMDP solver: backward induction over alpha-vectors.

Implements equations (1)-(3) and (6) of Appendix A. An alpha-vector is a
length-|S| array giving V_n(s) for one action; V_n(b) is the max over the
stage's alpha-vectors of alpha . b.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog

from pomdp_breast_cancer.pomdp import POMDP


@dataclass
class AlphaVector:
    values: np.ndarray
    action_idx: int


def _backup(pomdp: POMDP, prev_vectors: list[AlphaVector]) -> list[AlphaVector]:
    n_actions = len(pomdp.actions)
    n_obs = len(pomdp.observations)

    candidates = []
    for a in range(n_actions):
        # g[o][i](s) = sum_s' T(s'|s) Z(o|s',a) alpha_i(s'), the T/Z term of (6)
        g = [
            [pomdp.transition[a] @ (pomdp.observation[a, :, o] * v.values) for v in prev_vectors]
            for o in range(n_obs)
        ]
        for choice in product(range(len(prev_vectors)), repeat=n_obs):
            total = pomdp.reward[a].copy()
            for o, i in enumerate(choice):
                total = total + pomdp.discount * g[o][i]
            candidates.append(AlphaVector(total, action_idx=a))
    return candidates


def _has_witness_belief(candidate: AlphaVector, rivals: list[AlphaVector], n_states: int) -> bool:
    """Is there a b where candidate . b beats every rival . b?

    Solved as an LP over b: maximize eps subject to candidate . b >= rival . b + eps
    for every rival, sum(b) = 1, b >= 0.
    """
    if not rivals:
        return True
    objective = np.zeros(n_states + 1)
    objective[-1] = -1.0  # linprog minimizes; minimize -eps to maximize eps
    a_ub, b_ub = [], []
    for rival in rivals:
        diff = rival.values - candidate.values
        a_ub.append(np.concatenate([diff, [1.0]]))
        b_ub.append(0.0)
    a_eq = [np.concatenate([np.ones(n_states), [0.0]])]
    bounds = [(0.0, 1.0)] * n_states + [(None, None)]
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds)
    if not result.success:
        raise RuntimeError(f"LP failed while checking alpha-vector dominance: {result.message}")
    # numerical tolerance: treat a margin this close to zero as no real advantage
    return -result.fun > 1e-9


def _prune(vectors: list[AlphaVector], n_states: int) -> list[AlphaVector]:
    """Drop every alpha-vector with no witness belief, i.e. dominated everywhere on the simplex."""
    kept: list[AlphaVector] = []
    remaining = list(vectors)
    while remaining:
        candidate = remaining.pop()
        rivals = kept + remaining
        if _has_witness_belief(candidate, rivals, n_states):
            kept.append(candidate)
    return kept


def solve(pomdp: POMDP, horizon: int) -> list[list[AlphaVector]]:
    """Run backward induction for `horizon` stages.

    stages[n] holds the (pruned) alpha-vectors for V_n, the value function with
    n periods remaining; stages[0] is the zero base case.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n_states = len(pomdp.states)
    stages = [[AlphaVector(np.zeros(n_states), action_idx=0)]]
    for _ in range(horizon):
        candidates = _backup(pomdp, stages[-1])
        stages.append(_prune(candidates, n_states))
    return stages
