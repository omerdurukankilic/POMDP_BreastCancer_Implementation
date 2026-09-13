# POMDP solver + interactive sandbox — design spec

## Goal

The proposal's Timeline section claims "a working demonstration of the model
above, formulated, solved, and parameterized for breast cancer follow-up,
already exists." Right now that's only half true: `pomdp.py` formulates the
model and updates beliefs, but nothing solves it or parameterizes it with
real numbers. This spec closes that gap with the smallest build that makes
the claim honest, plus an interactive sandbox for exploring the result.

## Scope

**In scope:**
- An exact finite-horizon POMDP solver (backward induction over alpha-vectors).
- Two illustrative risk strata (low, high) instead of full PREDICT integration.
- A script that solves both strata and prints the visit-count/switch-point
  comparison described in the proposal's Data and Parameterization section.
- A static, self-contained HTML sandbox: pick a stratum, toggle between
  auto-simulated and manually-driven observations, step through the
  visits, watch the belief and recommended action update.

**Note on horizon length:** the proposal describes the model over 10
decision points (5 years, 6-month steps). The built demo solves and
steps through a 6-period horizon instead. This was discovered during
implementation: once the reward model has genuine decision-relevant
structure (`DETECTION_BENEFIT` making screening actually valuable), the
exact alpha-vector solver's pruned vector set grows combinatorially with
horizon length, and horizon=10 is no longer tractable in demo timeframes.
This is a deliberate, disclosed scope reduction for the demo, not a claim
that the proposal's 10-period design is wrong. See `scripts/run_demo.py`'s
`HORIZON` constant.

**Out of scope (not this round):**
- Real PREDICT integration or any patient-level data — none is needed; the
  model runs on transition/observation/reward numbers, not a dataset.
- More than two strata.
- Point-based approximate solvers (Perseus) or robust-POMDP methods — those
  are for later, larger extensions, not this demo.
- Live in-browser re-solving (Pyodide). The sandbox replays an
  already-solved policy; changing a parameter means re-running the Python
  script and re-exporting.

## Components

### `src/pomdp_breast_cancer/solver.py` (new)

Exact backward induction over alpha-vectors, matching Appendix A equations
(1)-(3) of the proposal. An alpha-vector is just a length-`|S|` array paired
with the action that owns it. At each stage, candidate vectors are built for
every action and every combination of "which vector from the previous
stage wins for each observation," then pruned down to the ones that are
actually optimal somewhere on the belief simplex. `scipy.optimize.linprog`
(already a dependency) does that pruning check — this is the real, textbook
algorithm, not a shortcut, since solving it correctly is what RQ1 is about.

Returns, for each stage `n = 0..N`, the set of `(alpha_vector, action)`
pairs representing `V_n`.

### `src/pomdp_breast_cancer/parameters.py` (new)

Builds two `POMDP` instances — low-risk and high-risk stratum — with:

- **States**: disease-free, occult loco-regional recurrence, occult distant
  recurrence, death from another cause.
- **Actions**: defer, standard, intensive.
- **Observations**: negative, positive.
- **Observation probabilities (Z)**: taken directly from the proposal's own
  verified Table 1 (Robertson et al. for `Intensive`, Kramer/Barton for
  `Standard`, Elmore/Otten for `Defer`).
- **Transition probabilities (T)**: illustrative hazard rates
  (`LOW_RISK`, `HIGH_RISK` in `parameters.py`), not sourced from a specific
  citation. See "Data honesty" below.
- **Reward (R)**: an illustrative QALY-shaped structure, three parts, none
  literature-derived: a per-state quality score (`STATE_QUALITY`, highest
  for disease-free, zero for death), a per-action cost (`ACTION_COST`,
  zero for defer, highest for intensive), and a detection-benefit term
  (`DETECTION_BENEFIT`) added in the two recurrence states, scaled by that
  action's sensitivity. The detection-benefit term was not in the original
  design; it was added after solving the model with just quality-minus-cost
  and finding `defer` won at every belief state, because nothing in that
  version ever rewarded actually catching a recurrence. This term is the
  concrete form of the proposal's own sentence describing the reward
  mechanism ("combining the benefit of detecting a recurrence early against
  the... cost and burden of the visit itself" — proposal §3.1), which the
  proposal states but does not give numbers for.

### `scripts/run_demo.py` (new)

Builds both strata, solves each, prints the same comparison the proposal's
prose describes (how many `Intensive`/`Standard` visits each stratum's
policy prescribes across the demo's decision points, and where the switch
point falls, if any). As noted under Scope above, the demo actually solves
a 6-period horizon rather than the proposal's 10, for solver tractability.
Exports the solved alpha-vectors and matrices as JSON,
embedded directly into the sandbox HTML file — no separate fetch, no CORS
issues, works as a plain static file.

### `sandbox/index.html` (new)

A self-contained page with two screens: a setup screen (stratum and
observation-mode choice, plus the data-honesty disclosure) and a simulation
screen (belief, recommended action, an expandable "why this action?"
breakdown, a always-visible previous-actions strip, and a step control).
Its JavaScript reimplements three small pure functions, not two as
originally scoped: the belief update and "which alpha-vector wins at this
belief" (both already covered by Python tests), plus a client-side
Bellman backup, `Q(b,a) = R(a).b + discount * sum_o P(o|b,a) * V_{n-1}(b^{a,o})`,
added to power the "why this action?" panel. This still isn't re-solving;
it evaluates the already-solved continuation value function (the previous
stage's pruned alpha-vectors) one step forward for each action, which is
exact regardless of whether that action's own vector survived pruning at
the current stage. The solver itself never runs in the browser.

## Data flow

Python solves once, offline → JSON embedded in the HTML → user picks a
stratum and a mode → steps through visits → JavaScript recomputes belief
and looks up the action from the exported data. Nothing calls back to
Python at runtime.

## Data honesty

This matters enough to state plainly, since it's the same standard the
proposal's own citations were held to all session: the detection
probabilities are real, verified numbers from the proposal's bibliography
(Table 1's sensitivity/specificity figures, midpointed where a range was
reported). The transition probabilities and reward values are not — they're
plausible, clearly-labeled placeholders standing in for what a real PREDICT
query and a real health-economic costing would supply. Code and output
should say so directly (a docstring or a printed note in the demo script),
not imply a sourcing that isn't there. The sandbox UI itself dropped its
own copy of this disclosure during the UI redesign; the requirement is
still met by `parameters.py`'s docstrings and the note `run_demo.py`
prints each time it solves and regenerates the sandbox.

Two further deviations from the proposal's own description, both
discovered during implementation rather than planned upfront, are worth
naming explicitly rather than leaving implicit in code comments:

- The demo solves a 6-period horizon, not the proposal's 10 (see "Note on
  horizon length" under Scope above).
- The reward function includes a `DETECTION_BENEFIT` term not spelled out
  anywhere in the proposal's text, added specifically to keep `defer` from
  dominating every belief state once the reward otherwise had real
  structure (see the `parameters.py` component section above). It gives
  concrete numeric form to the proposal's reward-mechanism sentence; it
  is not itself something the proposal specifies.

## Error handling

`POMDP.__post_init__` already validates array shapes; nothing new needed
there. The solver rejects a non-positive horizon. The sandbox mirrors the
existing zero-probability guard from `update_belief` and shows a plain
error if the embedded data is missing or malformed.

## Testing strategy (TDD)

Tests come first, same pattern as the existing `test_pomdp.py`:

- `tests/test_solver.py`: start with a 1-2 state POMDP small enough to
  solve by hand, then a brute-force cross-check against a small enumerated
  case, before touching the real 4-state model at all.
- `tests/test_parameters.py`: matrix shapes, probabilities summing to 1,
  and the exact sensitivity/specificity numbers pinned so a future edit
  can't silently drift from Table 1.
- An integration test solving both strata and asserting the high-risk
  policy recommends `Intensive` at least as early/often as low-risk — a
  concrete, checkable version of the RQ2 claim.

## Code conventions

- Humanize the code: write it the way a person thinking through the
  problem would, not in a defensive, over-engineered, or padded style.
  Plain names, straightforward control flow, no abstraction the current
  scope doesn't need.
- Do not overcomment. No comments restating what a line already says.
  A comment earns its place only when it explains something the code
  can't say for itself, e.g. why a hazard number was chosen, or why a
  pruning step is needed.
- When a comment does reference the math, use the notation Appendix A
  already defines, not a paraphrase. A reader flipping between the code
  and the proposal should see the same symbols in both places.

### Appendix A notation reference

| Symbol | Meaning |
|---|---|
| `S`, `s` | state set, a state |
| `A`, `a` | action set, an action |
| `Ω`, `o` | observation set, an observation |
| `T(s' \| s)` | transition probability (action-independent) |
| `Z(o \| s', a)` | observation probability |
| `R(s, a)` | reward |
| `γ` | discount factor |
| `N` | number of follow-up periods remaining |
| `b` | belief, `b ∈ Δ(S)` |
| `b^{a,o}` | belief after taking `a` and observing `o` |
| `η` | normalizer in the belief update |
| `V_n(b)` | value function at `n` periods remaining |
| `π_n(b)` | policy (action prescribed) at `n` periods remaining |

E.g. the solver's per-stage candidate-vector step is a direct
implementation of equation (2)/(6), so its docstring should say so in
those terms (`V_n`, `b^{a,o}`, `η`) rather than re-explaining the idea
from scratch in prose.
