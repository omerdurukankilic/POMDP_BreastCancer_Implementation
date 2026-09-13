"""States, actions, observations, and the transition/observation/reward matrices.

The detection numbers (sensitivity/specificity) come from the proposal's own
verified Table 1 — see `build_observation_matrix`. The transition hazard
rates (`LOW_RISK`, `HIGH_RISK`) and reward values (`STATE_QUALITY`,
`ACTION_COST`) below are illustrative placeholders, not sourced from any
citation — see the design spec's "Data honesty" section. Don't blur the two.
"""

from __future__ import annotations

import numpy as np

from pomdp_breast_cancer.pomdp import POMDP

STATES = ["disease_free", "loco_regional", "distant", "death_other"]
ACTIONS = ["defer", "standard", "intensive"]
OBSERVATIONS = ["negative", "positive"]

# Table 1's numbers: midpoints where a range was reported (intensive sens.
# 64-67%, standard sens. ~50-55%), single reported values otherwise.
DEFER_SENS, DEFER_SPEC = 0.25, 0.99
STANDARD_SENS, STANDARD_SPEC = 0.525, 0.94
INTENSIVE_SENS, INTENSIVE_SPEC = 0.655, 0.91


def _detection_row(sensitivity: float, specificity: float) -> np.ndarray:
    # Row order must match STATES: disease_free, loco_regional, distant,
    # death_other. disease_free and death_other have no recurrence to find,
    # so a "negative" reading there is governed by specificity;
    # loco_regional and distant do have recurrence present, so it's
    # governed by sensitivity. death_other is death from an unrelated
    # cause, so it's grouped with disease_free rather than getting its own
    # number.
    return np.array(
        [
            [specificity, 1 - specificity],
            [1 - sensitivity, sensitivity],
            [1 - sensitivity, sensitivity],
            [specificity, 1 - specificity],
        ]
    )


def build_observation_matrix() -> np.ndarray:
    """Build the (action, state, observation) detection probability matrix."""
    return np.array(
        [
            _detection_row(DEFER_SENS, DEFER_SPEC),
            _detection_row(STANDARD_SENS, STANDARD_SPEC),
            _detection_row(INTENSIVE_SENS, INTENSIVE_SPEC),
        ]
    )


# Illustrative placeholders (not sourced from PREDICT or any literature): a
# state's "quality" and each action's cost, combined below into R(s, a).
STATE_QUALITY = np.array([1.0, 0.8, 0.5, 0.0])
ACTION_COST = np.array([0.0, 0.01, 0.03])

# Illustrative placeholder, not sourced from any literature: the QALY-ish
# value of actually catching a recurrence at this visit. R(s, a) is an
# expectation over what the visit's observation reveals, so when a
# recurrence is present the sensitivity-weighted chance of detecting it
# belongs in the reward alongside the visit's cost — without it, screening
# has no benefit to offset ACTION_COST and `defer` dominates everywhere.
DETECTION_BENEFIT = 0.3

# Illustrative placeholders for annual transition hazards. Risk strata differ
# only in how fast disease progresses; the background non-cancer death rate
# is shared across strata.
BACKGROUND_DEATH_RATE = 0.002
LOW_RISK = {"to_loco_regional": 0.01, "to_distant": 0.005, "loco_to_distant": 0.05}
HIGH_RISK = {"to_loco_regional": 0.03, "to_distant": 0.015, "loco_to_distant": 0.10}


def _transition_matrix(
    to_loco_regional: float, to_distant: float, loco_to_distant: float
) -> np.ndarray:
    """Build T(s'|s), the same for every action, given a risk stratum's hazards."""
    t = np.zeros((len(STATES), len(STATES)))
    # Index by name, not position, so this stays correct if STATES is reordered
    # (matches the STATES.index(...) lookup build_reward_matrix uses below).
    disease_free = STATES.index("disease_free")
    loco_regional = STATES.index("loco_regional")
    distant = STATES.index("distant")
    death_other = STATES.index("death_other")

    t[disease_free, disease_free] = 1 - to_loco_regional - to_distant - BACKGROUND_DEATH_RATE
    t[disease_free, loco_regional] = to_loco_regional
    t[disease_free, distant] = to_distant
    t[disease_free, death_other] = BACKGROUND_DEATH_RATE

    t[loco_regional, loco_regional] = 1 - loco_to_distant - BACKGROUND_DEATH_RATE
    t[loco_regional, distant] = loco_to_distant
    t[loco_regional, death_other] = BACKGROUND_DEATH_RATE

    t[distant, distant] = 1 - BACKGROUND_DEATH_RATE
    t[distant, death_other] = BACKGROUND_DEATH_RATE

    # death_other is absorbing.
    t[death_other, death_other] = 1.0

    # A diagonal entry is 1 minus the sum of that row's hazards, so the row
    # summing to 1 doesn't by itself prove every entry is a valid probability
    # (hazards summing past 1 would drive the diagonal negative while the row
    # sum stayed exactly 1). Guard against that directly.
    assert (t >= 0).all(), "transition matrix has a negative entry — hazards sum to more than 1"
    return t


def build_reward_matrix() -> np.ndarray:
    """Build R(s, a): a state's quality minus the chosen action's cost, plus
    the expected benefit of detecting a recurrence that's actually present.

    STATE_QUALITY is broadcast across actions (rows) and ACTION_COST across
    states (columns), giving the (n_actions, n_states) shape POMDP expects.
    """
    reward = STATE_QUALITY[np.newaxis, :] - ACTION_COST[:, np.newaxis]
    # Death carries no ongoing quality-of-life or cost — zero it explicitly
    # rather than let it inherit STATE_QUALITY's placeholder 0.0 by accident.
    reward[:, STATES.index("death_other")] = 0.0

    # In a recurrence state, R(s, a) as an expectation over Z(o|s,a) picks up
    # a sensitivity-weighted DETECTION_BENEFIT term: a more sensitive action
    # is more likely to actually catch the recurrence this visit.
    sensitivities = {"defer": DEFER_SENS, "standard": STANDARD_SENS, "intensive": INTENSIVE_SENS}
    for a_idx, action in enumerate(ACTIONS):
        for state in ("loco_regional", "distant"):
            reward[a_idx, STATES.index(state)] += DETECTION_BENEFIT * sensitivities[action]

    return reward


def build_pomdp(risk: str) -> POMDP:
    """Assemble the full POMDP for a risk stratum ("low" or "high")."""
    hazards = {"low": LOW_RISK, "high": HIGH_RISK}.get(risk)
    if hazards is None:
        raise ValueError(f"unknown risk stratum: {risk!r}")

    # Transition doesn't depend on the action taken, so tile it across actions.
    single_stage_transition = _transition_matrix(**hazards)
    transition = np.tile(single_stage_transition, (len(ACTIONS), 1, 1))

    return POMDP(
        states=STATES,
        actions=ACTIONS,
        observations=OBSERVATIONS,
        transition=transition,
        observation=build_observation_matrix(),
        reward=build_reward_matrix(),
        discount=0.95,
    )
