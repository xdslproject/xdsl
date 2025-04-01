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
from xdsl.dialects.bigint import BigIntegerType, AddBigIntOp
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(int, BigIntegerType)
p.register_method(int, "__add__", AddBigIntOp)
with CodeContext(p):

    def foo(x: int, y: int) -> int:
        return x + y
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
  func.func() @foo(%0 : !bigint.bigint, %1 : !bigint.bigint) -> !bigint.bigint {
    %2 = bigint.add %0, %1 : !bigint.bigint
    func.return %2 : !bigint.bigint
  }
}
```
