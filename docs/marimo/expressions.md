
# Expressions in IR

```python {marimo}
import marimo as mo
# Uncomment the following two lines to install local version of xDSL.
# Adjust version string as required
import micropip
await micropip.install("xdsl @ http://127.0.0.1:8000/xdsl-0.0.0-py3-none-any.whl")
```

```python {marimo}
import xdsl
from xdsl.utils import marimo as xmo

expr_str = mo.ui.text(value = "1 + 2", debounce=1)
expr_str
```

Bla

```python {marimo}

expr = xmo.Expression.parse(expr_str.value if expr_str else "hello")

mo.md(f"Symbols: {expr.symbols}, {expr.expression}")
```
