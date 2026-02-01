# Code Style Guidelines

Follow these rules for all code changes in this repository:

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
- Centralize Control Flow: Branching logic belongs in parents. leaf functions should be
  pure logic.
- Fail Fast: Detect unexpected conditions immediately. Raise Exceptions rather than
  corrupt state.
- Ask for forgiveness than permission: assume the existence of valid keys or attributes
  and catch exceptions if the assumption proves false. Use try and except statements.

The test and validation suite consists of:

- Unit tests (pytest): Python tests in `tests/` (excluding `tests/filecheck`) for testing
  APIs and logic. Run with `make pytest`. Coverage can be checked with
  `make coverage && make coverage-report`.
- Integration tests (lit): File-based tests in `tests/filecheck` using `FileCheck` to
  verify tool output. Preferred for compiler passes and round-tripping. Run with
  `make filecheck`.
- Static type checking (pyright): Type checking of Python files to ensure type safety.
  Configured in `pyproject.toml`. Run with `make pyright`.

Before committing, make sure the following commands run successfully:

- `make tests`
- `make precommit`
