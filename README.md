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
├── tests/                     # unit tests
├── .github/workflows/         # CI configuration
├── pyproject.toml             # package metadata and dependencies
└── requirements.txt           # pinned runtime dependencies
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Status

Early scaffolding. Model definitions, solvers, and data loaders are
being implemented incrementally.

## License

MIT, see [LICENSE](LICENSE).
