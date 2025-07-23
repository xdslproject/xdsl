# RUN: python %s | filecheck %s

"""Tests for PDL rewriting Python DSL.

These tests exercise the API providing an evaluatable Python DSL for expressing
the semantics of rewrites, which can then be lowered to the PDL dialect using
the PyAST frontend.

The following code snippets show early suggestions for the structure of this API:

```python
# Derived from <https://github.com/xdslproject/xdsl/pull/1137>

@rewrite_pattern_query
def zero_to_one_query(root: arith.arith.ConstantOp):
    zero = builtin.IntegerAttr(0, 32)
    return root.value == zero

@query_rewrite_pattern(rewrite_pattern_query)
def zero_to_one(rewriter: PatternRewriter, root: arith.arith.ConstantOp):
    one = builtin.IntegerAttr(1, 32)
    rewriter.replace_matched_op(arith.arith.ConstantOp(one))
````

```python
# Derived from <https://github.com/opencompl/xdsl-smt/blob/main/xdsl_smt/semantics/llvm_semantics.py#L225>

class XOrSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        # Perform the addition
        res = rewriter.insert(smt_bv.XorOp(lhs, rhs)).res

        return ((res, None),)
```

This API should then implement the following simple PDL MLIR rewrite:

```mlir
%val = arith.constant 0 : i32

pdl.pattern : benefit(2) {
  %0 = pdl.type
  %1 = pdl.attribute = 0 : i32
  %2 = pdl.operation "arith.constant" {"value" = %1} -> (%0 : !pdl.type)
  pdl.rewrite %2 {
    %3 = pdl.attribute = 1 : i32
    %4 = pdl.operation "arith.constant" {"value" = %3} -> (%0 : !pdl.type)
    pdl.replace %2 with %4
  }
}
```
"""

from xdsl.dialects import arith, builtin, pdl
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def integer_attr_as_pdl_attribute(
    value: int | builtin.IntAttr, value_type: int
) -> pdl.AttributeOp:
    """Get a `pdl.attribute` for an xDSL `IntegerAttr`."""
    return pdl.AttributeOp(builtin.IntegerAttr(value, value_type))


def arith_constant_as_pdl_operation(value: SSAValue[pdl.Attribute]) -> pdl.OperationOp:
    """Get a `pdl.operation` for an `arith.constant` constructor."""
    return pdl.OperationOp(
        "arith.constant",
        attribute_value_names=(builtin.StringAttr("value"),),
        attribute_values=(value,),
    )


ctx = PyASTContext()
ctx.register_function(builtin.IntegerAttr, integer_attr_as_pdl_attribute)
ctx.register_function(arith.ConstantOp, arith_constant_as_pdl_operation)
ctx.register_function(PatternRewriter.replace_op, pdl.ReplaceOp)


@ctx.parse_program  # TODO: Change to `@ctx.pdl_rewrite` when implemented
def constant_replace(
    rewriter: PatternRewriter, matched_operation: arith.ConstantOp
) -> None:
    """Replace a constant operation with a constant i32 with value one."""
    new_attribute = builtin.IntegerAttr(1, 32)
    new_operation = arith.ConstantOp(new_attribute)
    rewriter.replace_op(matched_operation, new_operation)


# Check that the DSL correctly rewrites on the xDSL data structures
matched_operation = arith.ConstantOp(builtin.IntegerAttr(0, 32))
module = builtin.ModuleOp([matched_operation])
rewriter = PatternRewriter(matched_operation)

print(module)
# CHECK:       builtin.module {
# CHECK-NEXT:    %0 = arith.constant 0 : i32
# CHECK-NEXT:  }

constant_replace(rewriter, matched_operation)
print(module)
# CHECK:       builtin.module {
# CHECK-NEXT:    %0 = arith.constant 1 : i32
# CHECK-NEXT:  }

# Check that the extracted module results in the correct PDL rewrite
print(constant_replace.module)
# CHECK:       builtin.module {
# CHECK-NEXT:    pdl.rewrite %2 {
# CHECK-NEXT:        %3 = pdl.attribute = 1 : i32
# CHECK-NEXT:        %4 = pdl.operation "arith.constant" {"value" = %3} -> (%0 : !pdl.type)
# CHECK-NEXT:        pdl.replace %2 with %4
# CHECK-NEXT:      }
# CHECK-NEXT:  }
