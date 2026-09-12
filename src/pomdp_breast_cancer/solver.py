"""Exact finite-horizon POMDP solver: backward induction over alpha-vectors.

Implements equations (1)-(3) and (6) of Appendix A. An alpha-vector is a
length-|S| array giving V_n(s) for one action; V_n(b) is the max over the
stage's alpha-vectors of alpha . b.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

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


def solve(pomdp: POMDP, horizon: int) -> list[list[AlphaVector]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n_states = len(pomdp.states)
    stages = [[AlphaVector(np.zeros(n_states), action_idx=0)]]
    for _ in range(horizon):
        stages.append(_backup(pomdp, stages[-1]))
    return stages
