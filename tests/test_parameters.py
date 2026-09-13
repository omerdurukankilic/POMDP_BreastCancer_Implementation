import numpy as np
import pytest

from pomdp_breast_cancer.parameters import (
    ACTIONS,
    OBSERVATIONS,
    STATES,
    build_observation_matrix,
)


def test_state_action_observation_lists_match_the_proposal():
    assert STATES == ["disease_free", "loco_regional", "distant", "death_other"]
    assert ACTIONS == ["defer", "standard", "intensive"]
    assert OBSERVATIONS == ["negative", "positive"]


def test_observation_matrix_rows_sum_to_one():
    observation = build_observation_matrix()
    assert observation.shape == (len(ACTIONS), len(STATES), len(OBSERVATIONS))
    assert observation.sum(axis=2) == pytest.approx(np.ones((len(ACTIONS), len(STATES))))


def test_observation_matrix_uses_table_1_numbers():
    observation = build_observation_matrix()
    disease_free = STATES.index("disease_free")
    loco_regional = STATES.index("loco_regional")
    negative, positive = OBSERVATIONS.index("negative"), OBSERVATIONS.index("positive")

    # Table 1: Intensive sensitivity 64-67%, specificity 85-97% (midpoints used)
    intensive = ACTIONS.index("intensive")
    assert observation[intensive, disease_free, negative] == pytest.approx(0.91)
    assert observation[intensive, loco_regional, positive] == pytest.approx(0.655)

    # Table 1: Standard sensitivity ~52.5%, specificity 94%
    standard = ACTIONS.index("standard")
    assert observation[standard, disease_free, negative] == pytest.approx(0.94)
    assert observation[standard, loco_regional, positive] == pytest.approx(0.525)

    # Table 1: Defer sensitivity 25%, specificity 99%
    defer = ACTIONS.index("defer")
    assert observation[defer, disease_free, negative] == pytest.approx(0.99)
    assert observation[defer, loco_regional, positive] == pytest.approx(0.25)


def test_transition_rows_sum_to_one_and_death_is_absorbing():
    from pomdp_breast_cancer.parameters import build_pomdp

    for risk in ("low", "high"):
        pomdp = build_pomdp(risk)
        assert pomdp.transition.sum(axis=2) == pytest.approx(
            np.ones((len(ACTIONS), len(STATES)))
        )
        death_idx = STATES.index("death_other")
        assert pomdp.transition[0, death_idx, death_idx] == pytest.approx(1.0)


def test_high_risk_progresses_faster_than_low_risk():
    from pomdp_breast_cancer.parameters import build_pomdp

    low = build_pomdp("low")
    high = build_pomdp("high")
    disease_free = STATES.index("disease_free")
    loco_regional = STATES.index("loco_regional")
    distant = STATES.index("distant")

    assert (
        high.transition[0, disease_free, loco_regional]
        > low.transition[0, disease_free, loco_regional]
    )
    assert (
        high.transition[0, disease_free, distant] > low.transition[0, disease_free, distant]
    )
    assert (
        high.transition[0, loco_regional, distant] > low.transition[0, loco_regional, distant]
    )


def test_build_pomdp_rejects_unknown_stratum():
    from pomdp_breast_cancer.parameters import build_pomdp

    with pytest.raises(ValueError):
        build_pomdp("medium")
