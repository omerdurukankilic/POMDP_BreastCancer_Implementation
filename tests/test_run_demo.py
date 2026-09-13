import pytest
from scripts.run_demo import HORIZON, INITIAL_BELIEF, most_likely_trajectory

from pomdp_breast_cancer.parameters import build_pomdp
from pomdp_breast_cancer.solver import solve


@pytest.fixture(scope="module")
def low_solution():
    pomdp = build_pomdp("low")
    stages = solve(pomdp, HORIZON)
    trace = most_likely_trajectory(pomdp, stages, INITIAL_BELIEF)
    return pomdp, stages, trace


@pytest.fixture(scope="module")
def high_solution():
    pomdp = build_pomdp("high")
    stages = solve(pomdp, HORIZON)
    trace = most_likely_trajectory(pomdp, stages, INITIAL_BELIEF)
    return pomdp, stages, trace


def test_high_risk_starts_screening_no_later_than_low_risk(low_solution, high_solution):
    _, _, low_trace = low_solution
    _, _, high_trace = high_solution

    def first_screening_visit(trace):
        actions = [action for action, _ in trace]
        return next((i for i, a in enumerate(actions) if a != "defer"), len(actions))

    assert first_screening_visit(high_trace) <= first_screening_visit(low_trace)


def test_high_risk_screens_at_least_as_much_as_low_risk(low_solution, high_solution):
    _, _, low_trace = low_solution
    _, _, high_trace = high_solution

    def screening_visits(trace):
        return sum(1 for action, _ in trace if action != "defer")

    assert screening_visits(high_trace) >= screening_visits(low_trace)


def test_high_risk_screens_at_some_point(high_solution):
    # The two comparisons above (<=, >=) both hold trivially if neither
    # stratum ever screens at all, which is exactly the vacuous case that
    # motivated adding DETECTION_BENEFIT to the reward in the first place.
    # Pin down that high risk actually leaves `defer` at least once, so a
    # regression back to "defer always dominates" fails a test here instead
    # of only showing up as an unexciting printout from the demo script.
    _, _, high_trace = high_solution
    actions_taken = [action for action, _ in high_trace]
    assert any(action != "defer" for action in actions_taken)


def test_trajectory_has_one_entry_per_period(low_solution):
    _, _, low_trace = low_solution
    assert len(low_trace) == HORIZON
