# POMDP Solver + Demo Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the proposal's Timeline claim true — a working demo that formulates, solves, and parameterizes the breast-cancer follow-up POMDP for two risk strata, plus an interactive HTML sandbox for exploring the result.

**Architecture:** An exact finite-horizon POMDP solver (backward induction over alpha-vectors, LP-pruned) sits on top of the existing `POMDP` dataclass. A parameters module builds two fully-specified `POMDP` instances (low-risk, high-risk) from the proposal's verified detection numbers plus illustrative hazard/reward values. A demo script solves both, prints the comparison the proposal's prose describes, and bakes the solved result into a static HTML sandbox — no server, no build step, no in-browser Python.

**Tech Stack:** Python (numpy, scipy.optimize.linprog — both already dependencies), pytest, plain HTML/CSS/JS for the sandbox. Dependencies are managed with `uv`, not pip.

**Setup:** `uv sync` installs the project and its dev dependencies (reading `pyproject.toml`, no manual venv activation needed). Every command in this plan that runs Python or pytest should be run through `uv run` (e.g. `uv run pytest tests/test_solver.py -v`, `uv run python scripts/run_demo.py`) — this plan writes bare `pytest`/`python` in each step for brevity, but prefix each with `uv run` when actually executing.

Reference: see the design spec at `docs/superpowers/specs/2026-09-13-pomdp-solver-demo-design.md` and Appendix A of `proposal/proposal.tex` for the math notation used throughout (`T(s'|s)`, `Z(o|s',a)`, `R(s,a)`, `γ`, `b`, `b^{a,o}`, `η`, `V_n`, `π_n`).

---

### Task 1: Alpha-vector solver — base case and single-stage sanity

**Files:**
- Create: `src/pomdp_breast_cancer/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pomdp_breast_cancer.solver'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Exact finite-horizon POMDP solver: backward induction over alpha-vectors.

Implements equations (1)-(3) and (6) of Appendix A. An alpha-vector is a
length-|S| array giving V_n(s) for one action; V_n(b) is the max over the
stage's alpha-vectors of alpha . b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pomdp_breast_cancer.pomdp import POMDP


@dataclass
class AlphaVector:
    values: np.ndarray
    action_idx: int


def solve(pomdp: POMDP, horizon: int) -> list[list[AlphaVector]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n_states = len(pomdp.states)
    stages = [[AlphaVector(np.zeros(n_states), action_idx=0)]]
    for _ in range(horizon):
        stages.append(stages[-1])
    return stages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver.py -v`
Expected: FAIL on `test_solve_picks_the_better_action_when_state_never_changes` (the loop body is a stub) — this is expected, Task 2 replaces the stub. Confirm `test_solve_rejects_nonpositive_horizon` passes.

- [ ] **Step 5: Commit**

```bash
git add src/pomdp_breast_cancer/solver.py tests/test_solver.py
git commit -m "feat: add AlphaVector and solve() scaffold with horizon validation"
```

---

### Task 2: Backup step — generate candidate vectors for one stage

**Files:**
- Modify: `src/pomdp_breast_cancer/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver.py -v`
Expected: FAIL with `ImportError: cannot import name '_backup'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `solve`'s loop and add `_backup`:

```python
from itertools import product


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver.py -v`
Expected: PASS on the new test. `test_solve_picks_the_better_action_when_state_never_changes` now also passes since a single-vector backup with one action per stage produces exactly the right value (no pruning needed yet with only one previous vector, so this doesn't yet test pruning — that's Task 3).

- [ ] **Step 5: Commit**

```bash
git add src/pomdp_breast_cancer/solver.py tests/test_solver.py
git commit -m "feat: implement backward-induction backup step"
```

---

### Task 3: Prune dominated alpha-vectors via LP

**Files:**
- Modify: `src/pomdp_breast_cancer/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver.py -v`
Expected: FAIL with `ImportError: cannot import name '_prune'`

- [ ] **Step 3: Write minimal implementation**

```python
from scipy.optimize import linprog


def _has_witness_belief(candidate: AlphaVector, rivals: list[AlphaVector], n_states: int) -> bool:
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
    return bool(result.success and -result.fun > 1e-9)


def _prune(vectors: list[AlphaVector], n_states: int) -> list[AlphaVector]:
    kept: list[AlphaVector] = []
    remaining = list(vectors)
    while remaining:
        candidate = remaining.pop()
        rivals = kept + remaining
        if _has_witness_belief(candidate, rivals, n_states):
            kept.append(candidate)
    return kept
```

Then update `solve` to prune after each backup:

```python
def solve(pomdp: POMDP, horizon: int) -> list[list[AlphaVector]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n_states = len(pomdp.states)
    stages = [[AlphaVector(np.zeros(n_states), action_idx=0)]]
    for _ in range(horizon):
        candidates = _backup(pomdp, stages[-1])
        stages.append(_prune(candidates, n_states))
    return stages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver.py -v`
Expected: PASS on all tests so far.

- [ ] **Step 5: Commit**

```bash
git add src/pomdp_breast_cancer/solver.py tests/test_solver.py
git commit -m "feat: prune dominated alpha-vectors via LP witness search"
```

---

### Task 4: Cross-check the solver against a direct recursive reference

**Files:**
- Modify: `tests/test_solver.py`

This is the strongest correctness test: it recomputes `V_n(b)` directly from
equation (2), with no alpha-vectors at all, and checks it matches
`max(alpha.values @ b for alpha in stages[n])`.

- [ ] **Step 1: Write the failing test**

```python
def _direct_value(pomdp: POMDP, belief: np.ndarray, n: int) -> float:
    if n == 0:
        return 0.0
    best = -np.inf
    for a in range(len(pomdp.actions)):
        total = float(belief @ pomdp.reward[a])
        predicted = belief @ pomdp.transition[a]
        for o in range(len(pomdp.observations)):
            likelihood = pomdp.observation[a, :, o]
            unnormalized = predicted * likelihood
            p_o = unnormalized.sum()
            if p_o > 1e-12:
                next_belief = unnormalized / p_o
                total += pomdp.discount * p_o * _direct_value(pomdp, next_belief, n - 1)
        best = max(best, total)
    return best


def test_solver_matches_direct_recursive_value_at_several_beliefs():
    pomdp = POMDP(
        states=["healthy", "disease"],
        actions=["wait", "screen"],
        observations=["negative", "positive"],
        transition=np.array(
            [
                [[0.95, 0.05], [0.0, 1.0]],
                [[0.95, 0.05], [0.0, 1.0]],
            ]
        ),
        observation=np.array(
            [
                [[0.9, 0.1], [0.4, 0.6]],
                [[0.99, 0.01], [0.05, 0.95]],
            ]
        ),
        reward=np.array([[0.0, -10.0], [-1.0, -1.0]]),
        discount=0.9,
    )
    stages = solve(pomdp, horizon=2)

    for belief in [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.2, 0.8])]:
        for n in (1, 2):
            solver_value = max(v.values @ belief for v in stages[n])
            assert solver_value == pytest.approx(_direct_value(pomdp, belief, n), abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver.py -v`
Expected: passes immediately if Tasks 1-3 are correct, or fails with a
numeric mismatch if there's a bug in `_backup`/`_prune` — either way, run
it now before moving on, since this is the test that would actually catch
a solver bug the smaller unit tests miss.

- [ ] **Step 3: Fix implementation if needed**

If this fails, the bug is almost always an index mix-up between `s` and
`s'` in `_backup`'s use of `pomdp.transition[a]` and
`pomdp.observation[a, :, o]` — re-check against the shapes documented in
`pomdp.py`'s docstring (`transition[a, s, s']`, `observation[a, s', o]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_solver.py
git commit -m "test: cross-check solver against direct recursive value computation"
```

---

### Task 5: Parameters module — states, actions, observations, detection numbers

**Files:**
- Create: `src/pomdp_breast_cancer/parameters.py`
- Test: `tests/test_parameters.py`

- [ ] **Step 1: Write the failing test**

```python
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
    intensive = ACTIONS.index("intensive")
    disease_free = STATES.index("disease_free")
    loco_regional = STATES.index("loco_regional")
    negative, positive = OBSERVATIONS.index("negative"), OBSERVATIONS.index("positive")

    # Table 1: Intensive sensitivity 64-67%, specificity 85-97% (midpoints used)
    assert observation[intensive, disease_free, negative] == pytest.approx(0.91)
    assert observation[intensive, loco_regional, positive] == pytest.approx(0.655)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parameters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pomdp_breast_cancer.parameters'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Builds the two risk-stratum POMDP instances for the follow-up demo.

Detection numbers (sensitivity/specificity) come from the proposal's own
verified Table 1. Transition and reward numbers are illustrative — see the
design spec's "Data honesty" section.
"""

from __future__ import annotations

import numpy as np

from pomdp_breast_cancer.pomdp import POMDP

STATES = ["disease_free", "loco_regional", "distant", "death_other"]
ACTIONS = ["defer", "standard", "intensive"]
OBSERVATIONS = ["negative", "positive"]

# midpoints of Table 1's reported ranges
DEFER_SENS, DEFER_SPEC = 0.25, 0.99
STANDARD_SENS, STANDARD_SPEC = 0.525, 0.94
INTENSIVE_SENS, INTENSIVE_SPEC = 0.655, 0.91


def _detection_row(sensitivity: float, specificity: float) -> np.ndarray:
    return np.array(
        [
            [specificity, 1 - specificity],
            [1 - sensitivity, sensitivity],
            [1 - sensitivity, sensitivity],
            [specificity, 1 - specificity],
        ]
    )


def build_observation_matrix() -> np.ndarray:
    return np.array(
        [
            _detection_row(DEFER_SENS, DEFER_SPEC),
            _detection_row(STANDARD_SENS, STANDARD_SPEC),
            _detection_row(INTENSIVE_SENS, INTENSIVE_SPEC),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parameters.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pomdp_breast_cancer/parameters.py tests/test_parameters.py
git commit -m "feat: build observation matrix from proposal's Table 1 numbers"
```

---

### Task 6: Parameters module — transition, reward, and build_pomdp()

**Files:**
- Modify: `src/pomdp_breast_cancer/parameters.py`
- Test: `tests/test_parameters.py`

- [ ] **Step 1: Write the failing test**

```python
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

    assert (
        high.transition[0, disease_free, loco_regional]
        > low.transition[0, disease_free, loco_regional]
    )


def test_build_pomdp_rejects_unknown_stratum():
    from pomdp_breast_cancer.parameters import build_pomdp

    with pytest.raises(ValueError):
        build_pomdp("medium")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parameters.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_pomdp'`

- [ ] **Step 3: Write minimal implementation**

Append to `parameters.py`:

```python
STATE_QUALITY = np.array([1.0, 0.8, 0.5, 0.0])
ACTION_COST = np.array([0.0, 0.01, 0.03])

BACKGROUND_DEATH_RATE = 0.002
LOW_RISK = {"to_loco_regional": 0.01, "to_distant": 0.005, "loco_to_distant": 0.05}
HIGH_RISK = {"to_loco_regional": 0.03, "to_distant": 0.015, "loco_to_distant": 0.10}


def _transition_matrix(to_loco_regional: float, to_distant: float, loco_to_distant: float) -> np.ndarray:
    t = np.zeros((len(STATES), len(STATES)))
    disease_free, loco_regional, distant, death_other = range(len(STATES))

    t[disease_free, disease_free] = 1 - to_loco_regional - to_distant - BACKGROUND_DEATH_RATE
    t[disease_free, loco_regional] = to_loco_regional
    t[disease_free, distant] = to_distant
    t[disease_free, death_other] = BACKGROUND_DEATH_RATE

    t[loco_regional, loco_regional] = 1 - loco_to_distant - BACKGROUND_DEATH_RATE
    t[loco_regional, distant] = loco_to_distant
    t[loco_regional, death_other] = BACKGROUND_DEATH_RATE

    t[distant, distant] = 1 - BACKGROUND_DEATH_RATE
    t[distant, death_other] = BACKGROUND_DEATH_RATE

    t[death_other, death_other] = 1.0
    return t


def build_reward_matrix() -> np.ndarray:
    reward = STATE_QUALITY[np.newaxis, :] - ACTION_COST[:, np.newaxis]
    reward[:, STATES.index("death_other")] = 0.0
    return reward


def build_pomdp(risk: str) -> POMDP:
    hazards = {"low": LOW_RISK, "high": HIGH_RISK}.get(risk)
    if hazards is None:
        raise ValueError(f"unknown risk stratum: {risk!r}")

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parameters.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pomdp_breast_cancer/parameters.py tests/test_parameters.py
git commit -m "feat: add transition/reward matrices and build_pomdp()"
```

---

### Task 7: Demo script — solve both strata and print the comparison

**Files:**
- Create: `scripts/run_demo.py`
- Test: `tests/test_run_demo.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from pomdp_breast_cancer.parameters import build_pomdp
from pomdp_breast_cancer.solver import solve
from scripts.run_demo import HORIZON, INITIAL_BELIEF, most_likely_trajectory


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_demo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'` — add an
empty `scripts/__init__.py` so it's importable as a package, then it
should fail instead with `ImportError: cannot import name 'HORIZON'`.

```bash
touch scripts/__init__.py
```

- [ ] **Step 3: Write minimal implementation**

```python
"""Solve both risk strata and report how their policies differ."""

from __future__ import annotations

import numpy as np

from pomdp_breast_cancer.parameters import build_pomdp
from pomdp_breast_cancer.pomdp import POMDP
from pomdp_breast_cancer.solver import AlphaVector, solve

HORIZON = 10
INITIAL_BELIEF = np.array([0.95, 0.03, 0.01, 0.01])


def most_likely_trajectory(
    pomdp: POMDP, stages: list[list[AlphaVector]], initial_belief: np.ndarray
) -> list[tuple[str, str]]:
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
    switch_visit = next(
        (i + 1 for i, action in enumerate(actions_taken) if action == "intensive"), None
    )
    print(f"{risk} risk: {standard_count} standard visits, {intensive_count} intensive visits")
    if switch_visit:
        print(f"  switches to intensive at visit {switch_visit}")
    else:
        print("  never switches to intensive")


def main() -> None:
    for risk in ("low", "high"):
        pomdp = build_pomdp(risk)
        stages = solve(pomdp, HORIZON)
        trace = most_likely_trajectory(pomdp, stages, INITIAL_BELIEF)
        summarize(risk, pomdp, trace)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_demo.py -v`
Expected: PASS. Also run `python scripts/run_demo.py` directly and confirm
it prints a two-stratum comparison.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_demo.py scripts/__init__.py tests/test_run_demo.py
git commit -m "feat: add demo script that solves both strata and reports the comparison"
```

---

### Task 8: Export solved data into the sandbox template

**Files:**
- Create: `sandbox/template.html` (placeholder body only, filled in by Task 9)
- Modify: `scripts/run_demo.py`
- Test: `tests/test_run_demo.py`

- [ ] **Step 1: Write the failing test**

```python
import json


def test_export_data_is_json_serializable_and_has_expected_shape():
    from scripts.run_demo import export_data

    low_pomdp = build_pomdp("low")
    high_pomdp = build_pomdp("high")
    data = export_data(
        {
            "low": (low_pomdp, solve(low_pomdp, HORIZON)),
            "high": (high_pomdp, solve(high_pomdp, HORIZON)),
        }
    )

    serialized = json.dumps(data)
    reloaded = json.loads(serialized)
    assert set(reloaded.keys()) == {"low", "high"}
    assert reloaded["low"]["states"] == ["disease_free", "loco_regional", "distant", "death_other"]
    assert len(reloaded["low"]["stages"]) == HORIZON + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_demo.py -v`
Expected: FAIL with `ImportError: cannot import name 'export_data'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_demo.py`:

```python
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "sandbox" / "template.html"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "sandbox" / "index.html"


def export_data(strata_results: dict[str, tuple[POMDP, list[list[AlphaVector]]]]) -> dict:
    data = {}
    for risk, (pomdp, stages) in strata_results.items():
        data[risk] = {
            "states": pomdp.states,
            "actions": pomdp.actions,
            "observations": pomdp.observations,
            "transition": pomdp.transition.tolist(),
            "observation": pomdp.observation.tolist(),
            "stages": [
                [{"values": v.values.tolist(), "action": v.action_idx} for v in stage]
                for stage in stages
            ],
        }
    return data


def write_sandbox(data: dict) -> None:
    template = TEMPLATE_PATH.read_text()
    rendered = template.replace("__POMDP_DATA__", json.dumps(data))
    OUTPUT_PATH.write_text(rendered)
```

Update `main()` to build and write the sandbox:

```python
def main() -> None:
    strata_results = {}
    for risk in ("low", "high"):
        pomdp = build_pomdp(risk)
        stages = solve(pomdp, HORIZON)
        trace = most_likely_trajectory(pomdp, stages, INITIAL_BELIEF)
        summarize(risk, pomdp, trace)
        strata_results[risk] = (pomdp, stages)

    data = export_data(strata_results)
    data["initial_belief"] = INITIAL_BELIEF.tolist()
    write_sandbox(data)
    print(f"wrote {OUTPUT_PATH}")
```

Create a placeholder `sandbox/template.html` for now (Task 9 replaces the body):

```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>POMDP follow-up sandbox</title></head>
<body>
<script>const DATA = __POMDP_DATA__;</script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_demo.py -v`
Expected: PASS. Run `python scripts/run_demo.py` and confirm
`sandbox/index.html` is created and contains real JSON where
`__POMDP_DATA__` was.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_demo.py sandbox/template.html tests/test_run_demo.py
git commit -m "feat: export solved policies as JSON embedded in the sandbox HTML"
```

---

### Task 9: Build the interactive sandbox UI

**Files:**
- Modify: `sandbox/template.html`

This produces the actual sandbox page. No pytest coverage here — the logic
(belief update, alpha-vector argmax) mirrors what's already tested in
Python; verification is manual, in a browser.

- [ ] **Step 1: Write the full page**

Replace `sandbox/template.html` with:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>POMDP follow-up sandbox</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  .belief-bar { display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }
  .belief-bar .fill { background: #6c8ebf; height: 1rem; }
  .belief-bar .label { width: 12rem; }
  button { margin-right: 0.5rem; }
</style>
</head>
<body>
<h1>Follow-up policy sandbox</h1>

<label>Risk stratum:
  <select id="risk">
    <option value="low">Low</option>
    <option value="high">High</option>
  </select>
</label>
<br /><br />
<label>
  <input type="checkbox" id="auto" checked />
  Auto-simulate observations
</label>

<p id="status"></p>
<div id="belief"></div>
<div id="controls">
  <button id="next">Next visit</button>
  <button id="pick-negative" style="display:none">Observed: negative</button>
  <button id="pick-positive" style="display:none">Observed: positive</button>
</div>

<script>
const DATA = __POMDP_DATA__;
let state = null;

function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function bestAction(stageVectors, belief) {
  let best = stageVectors[0];
  let bestValue = dot(best.values, belief);
  for (const vector of stageVectors) {
    const value = dot(vector.values, belief);
    if (value > bestValue) {
      best = vector;
      bestValue = value;
    }
  }
  return best.action;
}

function predictedStateDistribution(strata, actionIdx, belief) {
  return strata.states.map((_, sPrime) =>
    belief.reduce((sum, b, s) => sum + b * strata.transition[actionIdx][s][sPrime], 0)
  );
}

function updateBelief(strata, actionIdx, obsIdx, belief) {
  const predicted = predictedStateDistribution(strata, actionIdx, belief);
  const unnormalized = predicted.map((p, sPrime) => p * strata.observation[actionIdx][sPrime][obsIdx]);
  const total = unnormalized.reduce((a, b) => a + b, 0);
  return unnormalized.map((v) => v / total);
}

function renderBelief(belief, states) {
  const container = document.getElementById("belief");
  container.innerHTML = "";
  states.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "belief-bar";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = name;
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${Math.round(belief[i] * 200)}px`;
    row.append(label, fill);
    container.appendChild(row);
  });
}

function startStratum() {
  const risk = document.getElementById("risk").value;
  const strata = DATA[risk];
  state = {
    strata,
    belief: DATA.initial_belief.slice(),
    periodsRemaining: strata.stages.length - 1,
  };
  renderBelief(state.belief, strata.states);
  document.getElementById("status").textContent = `${state.periodsRemaining} visits remaining`;
}

function step(obsIdx) {
  const { strata, belief, periodsRemaining } = state;
  if (periodsRemaining <= 0) return;

  const stageVectors = strata.stages[periodsRemaining];
  const actionIdx = bestAction(stageVectors, belief);
  const actionName = strata.actions[actionIdx];

  if (obsIdx === undefined) {
    const predicted = predictedStateDistribution(strata, actionIdx, belief);
    const obsProbs = strata.observations.map((_, o) =>
      predicted.reduce((sum, p, sPrime) => sum + p * strata.observation[actionIdx][sPrime][o], 0)
    );
    obsIdx = Math.random() < obsProbs[1] ? 1 : 0;
  }

  state.belief = updateBelief(strata, actionIdx, obsIdx, belief);
  state.periodsRemaining -= 1;
  renderBelief(state.belief, strata.states);
  document.getElementById("status").textContent =
    `visit action: ${actionName}, observed: ${strata.observations[obsIdx]}, ${state.periodsRemaining} visits remaining`;
}

document.getElementById("risk").addEventListener("change", startStratum);
document.getElementById("auto").addEventListener("change", () => {
  const manual = !document.getElementById("auto").checked;
  document.getElementById("pick-negative").style.display = manual ? "inline" : "none";
  document.getElementById("pick-positive").style.display = manual ? "inline" : "none";
  document.getElementById("next").style.display = manual ? "none" : "inline";
});
document.getElementById("next").addEventListener("click", () => step());
document.getElementById("pick-negative").addEventListener("click", () => step(0));
document.getElementById("pick-positive").addEventListener("click", () => step(1));

startStratum();
</script>
</body>
</html>
```

- [ ] **Step 2: Regenerate the sandbox and verify manually**

```bash
python scripts/run_demo.py
open sandbox/index.html
```

Check by hand:
- Switching the risk stratum dropdown resets the belief bars and visit counter.
- With "Auto-simulate" checked, "Next visit" alone drives the whole sequence.
- Unchecking "Auto-simulate" hides "Next visit" and shows the two observation buttons instead; clicking either advances one step using that observation.
- After 10 visits, further clicks do nothing (periodsRemaining reaches 0).

- [ ] **Step 3: Commit**

```bash
git add sandbox/template.html sandbox/index.html
git commit -m "feat: build interactive sandbox UI with auto/manual observation toggle"
```

---

### Task 10: Switch to uv and update the README

**Files:**
- Modify: `.pre-commit-config.yaml`
- Modify: `README.md`

- [ ] **Step 1: Make the pytest pre-commit hook use uv**

In `.pre-commit-config.yaml`, change:

```yaml
      - id: pytest
        name: pytest
        entry: pytest
        language: system
```

to:

```yaml
      - id: pytest
        name: pytest
        entry: uv run pytest
        language: system
```

This is why commits earlier in this project needed a manually-activated
venv for the hook to find `pytest` — routing it through `uv run` fixes
that for good, with no other setup needed.

- [ ] **Step 2: Replace the pip-based Getting started section with uv**

Replace the existing `## Getting started` section (currently
`python -m venv .venv` / `source .venv/bin/activate` / `pip install -e ".[dev]"`)
with:

```markdown
## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for dependency
management — no manual virtualenv activation or pip needed.

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```
```

- [ ] **Step 3: Update the Status and add a Demo section**

Replace the `## Status` section and add a new section after `## Getting started`:

```markdown
## Demo

Solve both risk strata and print the comparison:

```bash
uv run python scripts/run_demo.py
```

This also regenerates `sandbox/index.html`, a self-contained interactive
page — open it directly in a browser to step through a simulated patient's
follow-up, either auto-simulated or with manually-chosen observations.

## Status

Formulation, exact solving, and parameterization (two illustrative risk
strata) are implemented and tested. See
`docs/superpowers/specs/2026-09-13-pomdp-solver-demo-design.md` for what's
illustrative versus literature-sourced, and what's intentionally out of
scope for now (real PREDICT integration, more than two strata, robust-POMDP
methods).
```

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml README.md
git commit -m "chore: switch to uv for dependency management, document the demo"
```

---

## Self-review notes

- **Spec coverage:** solver (Tasks 1-4), parameters with the two strata (Tasks 5-6), demo script with the visit-count/switch-point comparison (Task 7), JSON export (Task 8), sandbox with auto/manual toggle (Task 9), data-honesty documentation carried into the README (Task 10). All spec sections have a corresponding task.
- **Placeholder scan:** no TBD/TODO; the one intentional stub (Task 1 Step 3) is explicitly called out as a stub the next task replaces, with a specific failing assertion named.
- **Type consistency:** `AlphaVector`, `solve`, `_backup`, `_prune`, `build_pomdp`, `most_likely_trajectory`, `export_data` are each defined once and used with the same signature everywhere they appear later.
