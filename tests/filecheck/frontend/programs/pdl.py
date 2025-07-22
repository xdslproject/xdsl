# RUN: python %s | filecheck %s

"""Tests for PDL rewriting Python DSL.

These tests exercise the API providing an evaluatable Python DSL for expressing
the semantics of rewrites, which can then be lowered to the PDL dialect using
the PyAST frontend.

The following code snippets show early suggestions for the structure of this API:

```python
# Derived from <https://github.com/xdslproject/xdsl/pull/1137>

@rewrite_pattern_query
def zero_to_one_query(root: arith.ConstantOp):
    zero = IntegerAttr(0, 32)
    return root.value == zero

@query_rewrite_pattern(rewrite_pattern_query)
def zero_to_one(rewriter: PatternRewriter, root: arith.ConstantOp):
    one = IntegerAttr(1, 32)
    rewriter.replace_matched_op(arith.ConstantOp(one))
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

print("Pass")
# CHECK: Pass
