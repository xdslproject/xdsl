# xDSL front-end: Embedding MLIR into Python

This document briefly describes the front-end framework. Please note that this
is an ongoing work, and that some things might not work as expected. Please
reach out if you find any inconsistency between this document and the code, or
if you spot bugs!

The goal of the front-end framework is to allow users to:

1. Write non-SSA xDSL/MLIR programs in Python (or Pythonic DSL).
2. Mix-in real Python code (limited functionality is supported).
3. Compile programs to xDSL (and subsequently to MLIR).

In the future, we plan to add dynamic execution of source programs as well.

## Your first toy program

To get started, first create a `FrontendProgram`, which you can compile or
transform later. Each snippet of code is encapsulated in `CodeContext` block.
Using a separate block for this allows to have a custom type checking for
Pythonic DSL types.

```python
# toy.py

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.pyast.context import CodeContext

p = FrontendProgram()
with CodeContext(p):

    # all your code will be here

    pass
```

Now we are ready to write a first simple program.

```python
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.pyast.context import CodeContext

# Pythonic DSL imports
from xdsl.frontend.pyast.dialects.builtin import i1, i32

p = FrontendProgram()
with CodeContext(p):

    def foo(x: i32, y: i32, z: i32) -> i32:
        return x + y * z
```

In order to be able to compile the program, you can simply call `compile()` on
your front-end program after the `CodeContext` block. Additionally, you can add
a print statement which will print out the generated xDSL.

```python
p.compile()
print(p.textual_format())
```

Finally, everything is set-up and so we can simply run `python toy.py`, which
should give the following output:

```mlir
builtin.module() {
  func.func() ["sym_name" = "foo", "function_type" = !fun<[!i32, !i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32, %2 : !i32):
    %3 : !i32 = arith.muli(%1 : !i32, %2 : !i32)
    %4 : !i32 = arith.addi(%0 : !i32, %3 : !i32)
    func.return(%4 : !i32)
  }
}
```

## Implementation notes

```
├── block.py                : Mark functions as basic blocks using a decorator.
├── code_generation.py      : Walk AST to generate xDSL from nodes. Simple
                              operands and control flow supported. Explicitly
                              no support for assignment. Implicitly no support
                              for complex structures such as classes
├── const.py                : Mark variables as compile-time constants using a
                              type hint.
├── context.py              : Context manager which parses its inner context
                              into a provided FrontendProgram, checking its
                              well-formedness on exit.
├── dialects
│   ├── arith.py            : Stubs and mappings for a subset of xDSL arith
│   └── builtin.py          : Stubs and mappings for a subset of xDSL builtin
├── exception.py            : Custom exceptions for the frontend and code 
                              generation
├── op_inserter.py          : Helper class to add operations from a stack to the
                              end of an operation/region/block
├── op_resolver.py          : Helper class to map frontend to xDSL operations
                              - `resolve_op` gets xDSL ops using "resolve_"
                                prefixed functions in dialects/
                              - `resolve_op_overload` ...?
├── passes
│   └── desymref.py         : Lower symref dialect into SSA form
├── program.py              : Helper class to store, compile and print the code
├── python_code_check.py    : Performs two checks for whether the code in the
                              context is supported:
                              1. Structure doesn't have nested block and blocks
                                 have explicit terminators
                              2. Guaranteeing and inlining constant values
├── symref.py               : Custom dialect to express typed variable semantics
                              like alloca but with symbols rather than memory
                              addresses
└── type_conversion.py      : Convert python type hints to xDSL types
                              - Caches conversions for performance
                              - `_convert_name` handles generic and concrete
                                type hint mappings into xDSL
                              - `convert_type_hint` wraps it with unimplemented features
```
