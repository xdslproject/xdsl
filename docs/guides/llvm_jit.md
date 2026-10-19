# JIT execution of LLVM IR modules

The `xdsl.jit.llvm.backend.LLVMJITBackend` class compiles xDSL modules written in the
LLVM dialect to native code, using llvmlite’s MCJIT execution engine and makes their
signatures callable from Python.

To achieve this, the backend first derives the C function signature from the IR and then
applies LLVM optimizations at level 2 by default.

For example, take the following `add.mlir` file, followed by a `jit.py` file which
compiles and executes it from the Python API:

```mlir
builtin.module {
  llvm.func @main() -> i32 {
    %left = llvm.mlir.constant(21 : i32) : i32
    %right = llvm.mlir.constant(21 : i32) : i32
    %result = llvm.add %left, %right : i32
    llvm.return %result : i32
  }
}
```

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["xdsl[llvm]"]
# ///
import sys
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects import builtin, llvm
from xdsl.jit.llvm.backend import LLVMJITBackend

ctx = Context()
ctx.load_dialect(builtin.Builtin)
ctx.load_dialect(llvm.LLVM)

with open(sys.argv[1]) as f:
    module = Parser(ctx, f.read()).parse_module()

backend = LLVMJITBackend(lowering=())
main = backend.jit(module, "main", ctx)

print(main.c_func())
```

```bash
uv run jit.py add.mlir  # 42
```

We pass `lowering=()` to LLVMJITBackend, because the module is already in the LLVM
dialect and doesn’t need additional lowering tooling.
