# RUN: python %s | filecheck %s

"""Tests for PDL rewriting Python DSL.

These tests exercise the API providing an evaluatable Python DSL for expressing
the semantics of rewrites, which can then be lowered to the PDL dialect using
the PyAST frontend.

The following code snippets show early suggestions for the structure of this API:

- <https://github.com/xdslproject/xdsl/pull/1137>
- <https://github.com/opencompl/xdsl-smt/blob/main/xdsl_smt/semantics/llvm_semantics.py#L225>

This example exercises the simple case of erasing a matched operation:

```mlir
pdl.pattern : benefit(2) {
  %0 = pdl.type
  %1 = pdl.attribute = 0 : i32
  %2 = pdl.operation "arith.constant" {"value" = %1} -> (%0 : !pdl.type)
  pdl.rewrite %2 {
    pdl.erase %2
  }
}
```
"""

from xdsl.dialects import arith, builtin, pdl
from xdsl.frontend import pypdl
from xdsl.ir import Operation
from xdsl.rewriter import Rewriter


def erase_op(operation: Operation) -> None:
    """Shim to avoid `Expr` AST node required for methods."""
    return Rewriter.erase_op(operation)


ctx = pypdl.PyPDLContext()
ctx.register_type(arith.ConstantOp, pdl.OperationType())
ctx.register_function(erase_op, pdl.EraseOp)


@ctx.parse_program
def constant_replace(matched_operation: arith.ConstantOp) -> arith.ConstantOp:
    erase_op(matched_operation)
    # NOTE: A value is returned due to limitations of PyAST's current function implementation
    return matched_operation


# Check that the DSL correctly rewrites on the xDSL data structures
matched_operation = arith.ConstantOp(builtin.IntegerAttr(0, 32))
module = builtin.ModuleOp([matched_operation])

print(module)
# CHECK:       builtin.module {
# CHECK-NEXT:    %0 = arith.constant 0 : i32
# CHECK-NEXT:  }

constant_replace(matched_operation)
print(module)
# CHECK:       builtin.module {
# CHECK-NEXT:  }

# Check that the extracted module results in the correct PDL rewrite
print(constant_replace.module)
# CHECK:       builtin.module {
# CHECK-NEXT:    pdl.pattern @constant_replace : benefit(1) {
# CHECK-NEXT:      %matched_operation = "test.pureop"() : () -> !pdl.operation
# CHECK-NEXT:      pdl.rewrite %matched_operation {
# CHECK-NEXT:        pdl.erase %matched_operation
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:  }
