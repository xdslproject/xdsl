# xDSL front-end: Embedding MLIR into Python

This document briefly describes the front-end framework. Please note that this
is an ongoing work, and that some things might not work as expected. Please
reach out if you find any inconsistency between this document and the code, or
if you spot bugs!

## Your first toy program

To get started, first create a `PyASTContext`, which encapsulates the mapping
between Python and IR types, operations, and literals.
For each type, method, and literal used in the program, we must specify a
mapping using `register_type`, `register_function`, and `register_literal`
respectively.
All function arguments and return types must be annotated.
Use `-> None` for functions that return no value.
An explicit `return` must be the final statement directly in the function body, early
returns are not supported.
Using this, we can write our first simple program:

```python
from xdsl.dialects.arith import AddfOp, ConstantOp, MulfOp
from xdsl.dialects.builtin import FloatAttr, f64
from xdsl.frontend.pyast.context import PyASTContext

ctx = PyASTContext()
ctx.register_type(float, f64)
ctx.register_function(float.__add__, AddfOp)
ctx.register_function(float.__mul__, MulfOp)
ctx.register_literal(float, lambda v: ConstantOp(FloatAttr(v, 64)))


@ctx.parse_program
def foo(x: float, y: float) -> float:
    return x + y * 5.0


print(foo.module)
```

Finally, everything is set-up and so we can simply run the above code, which
should give the following output:

```mlir
builtin.module {
  func.func @foo(%x: f64, %y: f64) -> f64 {
    %0 = arith.constant 5.000000e+00 : f64
    %1 = arith.mulf %y, %0 : f64
    %2 = arith.addf %x, %1 : f64
    func.return %2 : f64
  }
}
```

## Package structure and implementation details

```text
.
├── utils
│   ├── builder.py
│   ├── exceptions.py
│   ├── op_inserter.py
│   └── type_conversion.py
├── code_generation.py
├── context.py
├── lowering.py
├── program.py
└── README.md
```

- `utils/builder.py`:
  Builds a module from a function AST.
  It generates the module, lowers its `py` operations and verifies it.
  Then it applies the configured transformation passes.
- `utils/exceptions.py`:
  Custom exceptions for the frontend and code generation.
- `utils/op_inserter.py`:
  Adds operations from a stack to the end of an operation, region or block.
- `utils/type_conversion.py`:
  Maps between Python objects, AST nodes and xDSL IR for types and functions.
- `code_generation.py`:
  Walks the AST to generate `py` from its nodes, without the registered mappings.
  Simple operands are supported, with no control flow.
  Assignment is only to variables or tuples of them.
  There is implicitly no support for complex structures such as classes.
- `context.py`:
  Configures the mappings and transformations.
  Decorates functions with `parse_program`.
- `lowering.py`:
  Lowers `py` to the operations registered in the context.
- `program.py`:
  Wraps a function as a callable.
  Its IR is built lazily, cached, and accessible through `module`.

## Future development

1. Widen support for Python control flow structures
   - [ ] Correctly handle scope, rather than just individually ingesting single
         functions
   - [ ] Extend `Call` AST to support invoking `func.FuncOp`s as well as
         functions registered to map to operations
2. Add support for more Python functionality by implementing code generation for
   each AST node:
   - [x] `Assert`
   - [ ] `AnnAssign`
   - [x] `Assign` (to a variable or a tuple of variables)
   - [x] `BinOp`
   - [ ] `Break`
   - [x] `Call`
   - [x] `Compare`
   - [ ] `For`
   - [x] `FunctionDef`
   - [ ] `If`/`IfExp`
   - [x] `Pass`
   - [x] `Return`
   - [ ] `While`
   - [ ] ...
3. Add support for more Python builtin data types, for example:
   - [ ] `None`
   - [ ] `string`
   - [ ] `list`
   - [x] `tuple` (unpacked only)
   - [ ] `dict`
   - [ ] `set`
