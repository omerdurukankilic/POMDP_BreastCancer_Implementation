import numpy as np
import pytest

from pomdp_breast_cancer.pomdp import POMDP


def make_two_state_pomdp() -> POMDP:
    states = ["healthy", "disease"]
    actions = ["wait", "screen"]
    observations = ["negative", "positive"]

    transition = np.array(
        [
            [[0.95, 0.05], [0.0, 1.0]],
            [[0.95, 0.05], [0.0, 1.0]],
        ]
    )
    observation = np.array(
        [
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.99, 0.01], [0.05, 0.95]],
        ]
    )
    reward = np.array(
        [
            [0.0, -10.0],
            [-1.0, -1.0],
        ]
    )
    return POMDP(states, actions, observations, transition, observation, reward)


def test_initial_belief_is_uniform():
    pomdp = make_two_state_pomdp()
    belief = pomdp.initial_belief()
    assert belief == pytest.approx([0.5, 0.5])


def test_update_belief_shifts_toward_disease_on_positive_screen():
    pomdp = make_two_state_pomdp()
    belief = pomdp.initial_belief()
    action_idx = pomdp.actions.index("screen")
    obs_idx = pomdp.observations.index("positive")

    updated = pomdp.update_belief(belief, action_idx, obs_idx)

    assert updated.sum() == pytest.approx(1.0)
    assert updated[pomdp.states.index("disease")] > belief[pomdp.states.index("disease")]


def test_update_belief_rejects_impossible_observation():
    states = ["s0"]
    actions = ["a0"]
    observations = ["o0", "o1"]
    transition = np.array([[[1.0]]])
    observation = np.array([[[1.0, 0.0]]])
    reward = np.array([[0.0]])
    pomdp = POMDP(states, actions, observations, transition, observation, reward)

    with pytest.raises(ValueError):
        pomdp.update_belief(pomdp.initial_belief(), 0, 1)


def test_expected_reward():
    pomdp = make_two_state_pomdp()
    belief = np.array([0.7, 0.3])
    action_idx = pomdp.actions.index("wait")

    assert pomdp.expected_reward(belief, action_idx) == pytest.approx(0.7 * 0.0 + 0.3 * -10.0)
