"""States, actions, observations, and the observation (detection) matrix.

The detection numbers here (sensitivity/specificity) come from the proposal's
own verified Table 1. Transition and reward numbers are a separate, later
concern (illustrative placeholders) — see the design spec's "Data honesty"
section. Don't blur the two: this module only builds the real, sourced Z
matrix.
"""

from __future__ import annotations

import numpy as np

STATES = ["disease_free", "loco_regional", "distant", "death_other"]
ACTIONS = ["defer", "standard", "intensive"]
OBSERVATIONS = ["negative", "positive"]

# midpoints of Table 1's reported ranges
DEFER_SENS, DEFER_SPEC = 0.25, 0.99
STANDARD_SENS, STANDARD_SPEC = 0.525, 0.94
INTENSIVE_SENS, INTENSIVE_SPEC = 0.655, 0.91


def _detection_row(sensitivity: float, specificity: float) -> np.ndarray:
    # disease_free and death_other have no recurrence to find, so a
    # "negative" reading there is governed by specificity; loco_regional
    # and distant do have recurrence present, so it's governed by
    # sensitivity. death_other is death from an unrelated cause, so it's
    # grouped with disease_free rather than getting its own number.
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
