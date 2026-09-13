# Contributing

This project is developed test-first. No production code is written
without a failing test that demands it.

## Workflow

1. **Red** — write one small test for the behavior you want. Run it and
   confirm it fails for the reason you expect, not because of a typo or
   import error.
2. **Green** — write the minimal code that makes the test pass. Resist
   adding anything the test does not require.
3. **Refactor** — with the suite green, clean up names, remove
   duplication, extract helpers. Do not change behavior here.
4. Repeat for the next behavior.

A change without a test that failed first is not accepted, including
bug fixes: reproduce the bug as a failing test before fixing it.

## Running things

```bash
make install      # uv sync + dev deps + pre-commit hook
make test         # run the suite once
make test-watch   # rerun affected tests on save
make lint         # ruff
make coverage     # pytest with a coverage report
```

`make install` also registers a pre-commit hook that runs ruff and the
full test suite before each commit.

## Test style

- One behavior per test, named for that behavior.
- Prefer real objects over mocks; the POMDP model has no external
  dependencies, so this is usually free.
- Keep tests fast. Slow tests get skipped under pressure, which defeats
  the point.

## Coverage

CI enforces a minimum coverage threshold (`pyproject.toml`,
`[tool.coverage.report]`). Coverage is a floor, not a goal: a fully
covered line proves nothing about whether the test that covers it
would fail if the line were wrong. Write the test for what should
happen, not to turn a line green.
