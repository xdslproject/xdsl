# Expressions in IR

```python {marimo}
import marimo as mo
```

A small test-page to test mkdocs-marimo.

## Input

```python {marimo}
expr_str = mo.ui.text(value = "3 + 2", debounce=False)
expr_str
```

## Output

```python {marimo}

res = expr_str

mo.md(f"Expr String: <pre> {res} </pre>")
```
