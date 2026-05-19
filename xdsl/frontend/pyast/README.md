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
│   ├── block.py              : Mark functions as basic blocks using a decorator
│   ├── const.py              : Mark variables as compile-time constants using a
│   │                           type hint
│   ├── exceptions.py         : Custom exceptions for the frontend and code
│   │                           generation
│   ├── op_inserter.py        : Helper class to add operations from a stack to
│   │                           the end of an operation/region/block
│   ├── python_code_check.py  : Performs two checks for whether the code in the
│   │                           context is supported:
│   │                           1. Structure doesn't have nested block and
│   │                              blocks have explicit terminators
│   │                           2. Guaranteeing and inlining constant values
│   └── type_conversion.py    : Map between Python objects, Python AST nodes,
│                               and xDSL IR for the types and functions
├── code_generation.py        : Walk AST to generate xDSL from nodes. Simple
│                               operands and control flow supported. Explicitly
│                               no support for assignment. Implicitly no support
│                               for complex structures such as classes
├── context.py                : Context manager which parses its inner context
│                               into a provided FrontendProgram, checking its
│                               well-formedness on exit.
├── program.py                : Helper class to store, compile, and print code
└── README.md
```

## Future development

1. Widen support for Python control flow structures
   - [ ] Correctly handle scope, rather than just individually ingesting single
         functions
   - [ ] Extend `Call` AST to support invoking `func.FuncOp`s as well as
         functions registered to map to operations
   - [ ] Loosen constraints enforced by `PythonCodeCheck` to leverage added
         functionality
2. Add support for more Python functionality by implementing code generation for
   each AST node:
   - [x] `Assert`
   - [ ] `AnnAssign`/`Assign`
   - [x] `BinOp`
   - [ ] `Break`
   - [x] `Call`
   - [x] `Compare`
   - [ ] `For`
   - [x] `FunctionDef`
   - [x] `If`/`IfExp`
   - [x] `Pass`
   - [x] `Return`
   - [ ] `While`
   - [ ] ...
3. Add support for more Python builtin data types, for example:
   - [ ] `None`
   - [ ] `string`
   - [ ] `list`
   - [ ] `tuple`
   - [ ] `dict`
   - [ ] `set`
