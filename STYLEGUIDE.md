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
  Assume valid keys or attributes exist and catch exceptions if the assumption proves
  false. Use try-except blocks:

  ```python
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
