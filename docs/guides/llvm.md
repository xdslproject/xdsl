# Exporting LLVM IR

The `xdsl.backend.llvm.convert.convert_module()` function maps the xDSL module to an
`llvmlite.ir.Module` and links it to the CLI command `xdsl-opt -t llvm`.

Using `llvm-diff` we made sure our emitted LLVM IR is semantically equivalent to
`mlir-translate`.

For example, take the following `hello.mlir` file:

```mlir
builtin.module {
  llvm.mlir.global internal constant @hello_str("Hello, World!\0A\00") {addr_space = 0 : i32} : !llvm.array<15 x i8>

  llvm.func @puts(!llvm.ptr)

  llvm.func @main() -> i32 {
    %0 = llvm.mlir.addressof @hello_str : !llvm.ptr
    llvm.call @puts(%0) : (!llvm.ptr) -> ()
    %ret = llvm.mlir.constant(0 : i32) : i32
    llvm.return %ret : i32
  }
}
```

```bash
xdsl-opt -t llvm hello.mlir | lli  # Hello, World!
```

The same conversion is available from the Python API, with the following `hello.py`
file:

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

print(convert_module(module, fallback_target_triple=None))
```

```bash
uv run hello.py hello.mlir | lli  # Hello, World!
```
