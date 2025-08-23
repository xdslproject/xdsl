# Expression Test Theo


```python {marimo}
import marimo as mo

def to_mlir2(stra):
    return str(len(stra))

expr_str = mo.ui.text(value = "1 + 2", debounce=False)
expr_str

mo.md(f"Expr String: <pre>{to_mlir2(str(expr_str))}</pre>")

#mo.md(f"Expr String: <pre>{to_mlir(expr_str.value)}</pre>")
```
