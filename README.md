# POMDP Breast Cancer Implementation

A Partially Observable Markov Decision Process (POMDP) framework for
modeling sequential decision making in breast cancer screening and
treatment, where the true disease state is not directly observable and
must be inferred from noisy clinical observations.

## Overview

Clinical decisions in breast cancer care, such as whether to screen,
biopsy, or treat, are made under uncertainty about the patient's true
underlying condition. A POMDP formalizes this setting as a tuple of
states, actions, observations, transition probabilities, observation
probabilities, and rewards, and produces a policy that chooses actions
to maximize expected long-term outcomes given a belief distribution
over hidden states.

This repository implements:

- A POMDP environment for breast cancer state modeling (disease
  progression, screening, and treatment actions).
- Belief state tracking and updates via Bayesian filtering.
- Solvers for computing or approximating optimal policies.
- Evaluation utilities for comparing policies against baselines.

## Project layout

```
POMDP_BreastCancer_Implementation/
├── src/
│   └── pomdp_breast_cancer/   # library code
├── scripts/
│   └── run_demo.py            # solves both risk strata and regenerates the sandbox
├── sandbox/
│   ├── template.html          # sandbox page template
│   └── index.html             # generated, committed interactive demo page
├── tests/                     # unit tests
├── .github/workflows/         # CI configuration
├── pyproject.toml             # package metadata and dependencies
└── uv.lock                    # pinned dependency versions
```

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for dependency
management, so no manual virtualenv activation or pip is needed.

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run pytest
```

## Demo

Solve both risk strata and print the comparison:

```bash
uv run python scripts/run_demo.py
```

This also regenerates `sandbox/index.html`, a self-contained interactive
page that can be opened directly in a browser to step through a simulated
patient's follow-up, either auto-simulated or with manually-chosen
observations. Re-run this script after any change to `parameters.py`,
`solver.py`, or `sandbox/template.html`, since `sandbox/index.html` is a
committed, generated file and will otherwise silently drift out of date.

## Status

Formulation, exact solving, and parameterization (two illustrative risk
strata) are implemented and tested. The demo differs from the proposal's
design in a few disclosed ways:

- Solves a 6-period horizon instead of the proposal's 10-period (5-year,
  6-month-step) design, since the exact solver's pruned vector set becomes
  intractable at 10 periods once the reward has real decision-relevant
  structure.
- Adds a detection-benefit term to the reward, scaled by the acting
  action's sensitivity, so that catching a recurrence actually outweighs
  a screening visit's cost.
- Uses two illustrative risk strata (low, high) rather than the full set
  PREDICT would distinguish.
- Parameterizes transition hazards and reward values with illustrative
  placeholders rather than numbers drawn from PREDICT or a real
  health-economic costing; only the detection sensitivities and
  specificities are literature-sourced (see `parameters.py`).

Real PREDICT integration, more than two strata, robust-POMDP methods, and
live in-browser re-solving are intentionally out of scope for now.

## License

MIT, see [LICENSE](LICENSE).
