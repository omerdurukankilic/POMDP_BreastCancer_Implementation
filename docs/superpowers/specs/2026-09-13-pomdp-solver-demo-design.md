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
  auto-simulated and manually-driven observations, step through the 10
  visits, watch the belief and recommended action update.

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
- **Transition probabilities (T)**: illustrative hazard rates, not sourced
  from a specific citation. See "Data honesty" below.
- **Reward (R)**: an illustrative QALY-shaped cost structure — visits cost
  more the more intensive they are, undetected progression costs more the
  further it goes. Also illustrative, not literature-derived.

### `scripts/run_demo.py` (new)

Builds both strata, solves each, prints the same comparison the proposal's
prose describes (how many `Intensive`/`Standard` visits each stratum's
policy prescribes across the 10 decision points, and where the switch point
falls, if any). Exports the solved alpha-vectors and matrices as JSON,
embedded directly into the sandbox HTML file — no separate fetch, no CORS
issues, works as a plain static file.

### `sandbox/index.html` (new)

A single self-contained page. Its JavaScript reimplements exactly two small
pure functions already covered by Python tests — the belief update and
"which alpha-vector wins at this belief" — nothing else. The solver itself
never runs in the browser.

UI: stratum picker, auto/manual toggle, a "next visit" control, a belief
display, the recommended action, and (in manual mode) two buttons to pick
the observation yourself.

## Data flow

Python solves once, offline → JSON embedded in the HTML → user picks a
stratum and a mode → steps through visits → JavaScript recomputes belief
and looks up the action from the exported data. Nothing calls back to
Python at runtime.

## Data honesty

This matters enough to state plainly, since it's the same standard the
proposal's own citations were held to all session: the detection
probabilities are real, verified numbers from the proposal's bibliography.
The transition probabilities and reward values are not — they're plausible,
clearly-labeled placeholders standing in for what a real PREDICT query and
a real health-economic costing would supply. Code and output should say so
directly (a docstring or a printed note in the demo script), not imply a
sourcing that isn't there.

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
