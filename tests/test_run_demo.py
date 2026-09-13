from scripts.run_demo import HORIZON, INITIAL_BELIEF, most_likely_trajectory

from pomdp_breast_cancer.parameters import build_pomdp
from pomdp_breast_cancer.solver import solve


def test_high_risk_switches_to_intensive_no_later_than_low_risk():
    low_pomdp = build_pomdp("low")
    high_pomdp = build_pomdp("high")
    low_stages = solve(low_pomdp, HORIZON)
    high_stages = solve(high_pomdp, HORIZON)

    low_trace = most_likely_trajectory(low_pomdp, low_stages, INITIAL_BELIEF)
    high_trace = most_likely_trajectory(high_pomdp, high_stages, INITIAL_BELIEF)

    def first_intensive_visit(trace):
        actions = [action for action, _ in trace]
        return next((i for i, a in enumerate(actions) if a == "intensive"), len(actions))

    assert first_intensive_visit(high_trace) <= first_intensive_visit(low_trace)


def test_trajectory_has_one_entry_per_period():
    pomdp = build_pomdp("low")
    stages = solve(pomdp, HORIZON)
    trace = most_likely_trajectory(pomdp, stages, INITIAL_BELIEF)
    assert len(trace) == HORIZON
