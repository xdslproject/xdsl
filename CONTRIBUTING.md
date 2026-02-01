# Contributing Guidelines

## Code Style

We aim to follow these rules for all changes in this repository:

- Match existing code style and architectural patterns.
- Zero Technical Debt: Fix issues immediately. Never rely on future refactoring.
- Keep it simple: No code > Obvious code > Clever code. Do not abstract prematurely.
- Locality over DRY: Prioritize code locality. Keep related logic close together even if
  it results in slight duplication. Prefer lambdas/inline logic over tiny single-use
  functions (<5 LoC). Minimize variable scope.
- Self-Describing Code: Minimize comments. Use descriptive variable names and constant
  intermediary variables to explain where possible.
- Guard-First Logic: Handle edge cases, invalid inputs and errors at the start of
  functions. Return early to keep the "happy path" at the lowest indentation level.
- Flat Structure: Keep if/else blocks small. Avoid nesting beyond two levels if possible.
- Centralize Control Flow: Branching logic belongs in parents. Leaf functions should be
  pure logic.
- Fail Fast: Detect unexpected conditions immediately. Raise Exceptions rather than
  corrupt state.
- [Ask for forgiveness not permission](https://docs.python.org/3/glossary.html#term-eafp):
  assume the existence of valid keys or attributes and catch exceptions if the assumption
  proves false. Use try-except blocks.

  ```Python
  # Good
  try:
      return mapping[key]
  except KeyError:
      return default_value

  # Bad
  if key in mapping:
      return mapping[key]
  return default_value
  ```

## Linting

We use the following tools to ensure code quality. Configuration for these tools can be
found in `pyproject.toml`.

- Ruff: Used for linting and formatting. Configured in `[tool.ruff]`.
- Pyright: Used for static type checking. Configured in `[tool.pyright]`.
  Run with `make pyright`.

## Testing

The test and validation suite consists of:

- Unit tests (pytest): Python tests in `tests/` (excluding `tests/filecheck`) for testing
  APIs and logic. Run with `make pytest`. Coverage can be checked with
  `make coverage && make coverage-report`.
- IR and Transformation tests (lit): File-based tests in `tests/filecheck` using
  `filecheck` (a Python reimplementation of LLVM's FileCheck) to verify tool output.
  These tests rely on the textual format to represent and construct IR. They are used to
  test that custom format implementations print and parse in the expected way, and to
  verify transformations such as pattern rewrites or passes. Run with
  `make filecheck`.

## Pre-commit Checklist

To ensure code quality, please set up the git hooks to run checks automatically:

```bash
make precommit-install
```

If you prefer to run checks manually, or to verify before committing, run:

- `make precommit` (for linting and formatting)
- `make tests` (for logic and type safety)
