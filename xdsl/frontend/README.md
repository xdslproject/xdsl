# xDSL front-end: Embedding MLIR into Python

This document briefly describes the front-end framework. Please not that this is
an onging work, and some things might not work as expected. Please reach out if
you find inconsistencies between this document and the code, or if you spot any
bugs.

The goal of the front-end framework is to allow users to

1. Write non-SSA xDSL/MLIR programs in Python (or Pythonic DSL).
2. Mix in real Python code (limited functionality).
3. Compile programs to xDSL (and MLIR).

In the future, we plan to add dynamic execution of source programs. 

## Your first toy program

To get started, first create a `FrontendProgram`, which you can compile or transform later.
Each snippet of code is encapsulated in `CodeContext` block. Using a separate block for this
allows to have a custom type checking for Pythonic DSL types.

```python
# toy.py

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    
    # all your code will be here

    pass
```

Now we are ready to write a first simple program.

```python
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

# Pythonic DSL imports
from xdsl.frontend.dialects.builtin import i1, i32

p = FrontendProgram()
with CodeContext(p):

    def foo(x: i32, y: i32, z: i32) -> i32:
        return x + y * z
```

In order to be able to compile the program, you can simply add the following
after the `CodeContext` block. Additionally, in this example we want to add a
print statement which will print out the generated xDSL.

```python
p.compile()
print(p.xdsl())
```

Finally, all is set-up and we can just run `python toy.py`, which should give
the following output.

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
