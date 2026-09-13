"""Solve both risk strata and report how their policies differ.

The transition hazards and reward values in `parameters.py` are illustrative
placeholders, not sourced from PREDICT or any citation; only the detection
sensitivities/specificities are real numbers from the proposal's
bibliography. This script's output should be read as a demonstration of the
model's mechanics, not as a clinical recommendation.
"""

from __future__ import annotations

import numpy as np

from pomdp_breast_cancer.parameters import build_pomdp
from pomdp_breast_cancer.pomdp import POMDP
from pomdp_breast_cancer.solver import AlphaVector, solve

# Reduced from the proposal's 10-period design: once the reward model
# gives screening genuine value (see parameters.DETECTION_BENEFIT), the
# exact alpha-vector solver's pruned vector count grows combinatorially
# with horizon length, and horizon=10 is no longer tractable in demo
# timeframes. This is a deliberate, documented scope reduction for the
# demo, not a claim about the proposal's own horizon.
HORIZON = 6
# Starting belief: mostly disease-free, with small mass on recurrence/death
# states to keep every observation reachable during the walk-forward below.
INITIAL_BELIEF = np.array([0.95, 0.03, 0.01, 0.01])


def most_likely_trajectory(
    pomdp: POMDP, stages: list[list[AlphaVector]], initial_belief: np.ndarray
) -> list[tuple[str, str]]:
    """Walk forward through the solved policy, always taking the most likely observation.

    At each visit, pick the action whose alpha-vector maximizes V_n(b) = alpha . b
    for the current belief b, then advance b via the observation the model
    itself considers most probable (argmax of Z(o|s',a) predicted forward),
    rather than sampling one. This gives one reproducible "typical" run for
    the console summary; it's deliberately simpler than the sandbox's
    random/manual observation modes.
    """
    belief = initial_belief
    horizon = len(stages) - 1
    trace = []
    for periods_remaining in range(horizon, 0, -1):
        vectors = stages[periods_remaining]
        best = max(vectors, key=lambda v: v.values @ belief)
        action_idx = best.action_idx

        predicted = belief @ pomdp.transition[action_idx]
        obs_probs = predicted @ pomdp.observation[action_idx]
        obs_idx = int(np.argmax(obs_probs))

        belief = pomdp.update_belief(belief, action_idx, obs_idx)
        trace.append((pomdp.actions[action_idx], pomdp.observations[obs_idx]))
    return trace


def summarize(risk: str, pomdp: POMDP, trace: list[tuple[str, str]]) -> None:
    actions_taken = [action for action, _ in trace]
    standard_count = actions_taken.count("standard")
    intensive_count = actions_taken.count("intensive")
    switch_visit, switch_action = next(
        ((i + 1, action) for i, action in enumerate(actions_taken) if action != "defer"),
        (None, None),
    )
    print(f"{risk} risk: {standard_count} standard visits, {intensive_count} intensive visits")
    if switch_visit:
        print(f"  switches to {switch_action} at visit {switch_visit}")
    else:
        print("  never leaves defer")


def main() -> None:
    print("Note: transition hazards and rewards are illustrative placeholders,")
    print("not sourced from PREDICT or the literature; only the detection")
    print("sensitivities/specificities are real. See parameters.py.\n")
    print("Solving both strata (this can take under a minute)...\n")
    for risk in ("low", "high"):
        pomdp = build_pomdp(risk)
        stages = solve(pomdp, HORIZON)
        trace = most_likely_trajectory(pomdp, stages, INITIAL_BELIEF)
        summarize(risk, pomdp, trace)


if __name__ == "__main__":
    main()
