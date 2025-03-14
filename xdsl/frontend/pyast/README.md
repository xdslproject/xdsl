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
