# Exporting LLVM IR

The `xdsl.backend.llvm.convert.convert_module()` function maps the xDSL module to
an `llvmlite.ir.Module` and links it to the CLI command `xdsl-opt -t llvm`.

Using `llvm-diff` we made sure our emitted LLVM IR is semantically equivalent to
`mlir-translate` for the operations covered by our conversion tests.

The input must contain only supported LLVM functions at the module's top level.

```bash
xdsl-opt -t llvm input.mlir
```

The same conversion is available from the Python API, with the following
`hello.py` file:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["xdsl[llvm]"]
# ///
import sys

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects import get_all_dialects
from xdsl.backend.llvm.convert import convert_module

ctx = Context()

for name, factory in get_all_dialects().items():
    ctx.register_dialect(name, factory)

with open(sys.argv[1]) as f:
    module = Parser(ctx, f.read()).parse_module()

print(convert_module(module))
```

```bash
uv run hello.py input.mlir
```
