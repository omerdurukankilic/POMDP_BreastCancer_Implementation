import numpy as np
import pytest

from pomdp_breast_cancer.pomdp import POMDP
from pomdp_breast_cancer.solver import solve


def make_one_state_pomdp(discount: float = 0.5) -> POMDP:
    return POMDP(
        states=["s0"],
        actions=["low_reward", "high_reward"],
        observations=["o0"],
        transition=np.array([[[1.0]], [[1.0]]]),
        observation=np.array([[[1.0]], [[1.0]]]),
        reward=np.array([[1.0], [5.0]]),
        discount=discount,
    )


def test_solve_rejects_nonpositive_horizon():
    pomdp = make_one_state_pomdp()
    with pytest.raises(ValueError):
        solve(pomdp, 0)


def test_solve_picks_the_better_action_when_state_never_changes():
    pomdp = make_one_state_pomdp()
    stages = solve(pomdp, horizon=1)

    assert len(stages) == 2
    assert stages[0][0].values == pytest.approx([0.0])

    best = max(stages[1], key=lambda v: v.values[0])
    assert best.action_idx == pomdp.actions.index("high_reward")
    assert best.values == pytest.approx([5.0])


def test_prune_removes_a_vector_with_no_witness_belief():
    from pomdp_breast_cancer.solver import AlphaVector, _prune

    dominated = AlphaVector(np.array([1.0, 1.0]), action_idx=0)
    dominator = AlphaVector(np.array([2.0, 2.0]), action_idx=1)

    kept = _prune([dominated, dominator], n_states=2)

    assert len(kept) == 1
    assert kept[0].action_idx == 1


def test_prune_keeps_two_vectors_that_cross():
    from pomdp_breast_cancer.solver import AlphaVector, _prune

    # crosses at b = [0.5, 0.5]: both vectors give value 1.5 there, but
    # each wins on one side of that point, so both are useful.
    left_winner = AlphaVector(np.array([2.0, 1.0]), action_idx=0)
    right_winner = AlphaVector(np.array([1.0, 2.0]), action_idx=1)

    kept = _prune([left_winner, right_winner], n_states=2)

    assert len(kept) == 2


def test_backup_produces_one_candidate_per_action_choice_combination():
    from pomdp_breast_cancer.solver import AlphaVector, _backup

    pomdp = POMDP(
        states=["s0", "s1"],
        actions=["a0", "a1"],
        observations=["o0", "o1"],
        transition=np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.9, 0.1], [0.2, 0.8]],
            ]
        ),
        observation=np.array(
            [
                [[0.7, 0.3], [0.4, 0.6]],
                [[0.7, 0.3], [0.4, 0.6]],
            ]
        ),
        reward=np.array([[1.0, 0.0], [0.0, 1.0]]),
        discount=1.0,
    )
    prev_vectors = [AlphaVector(np.array([0.0, 0.0]), action_idx=0)]

    candidates = _backup(pomdp, prev_vectors)

    # 2 actions * 1 vector-choice-per-observation ^ 2 observations = 2 candidates
    assert len(candidates) == 2
    for candidate in candidates:
        expected = pomdp.reward[candidate.action_idx]
        assert candidate.values == pytest.approx(expected)
