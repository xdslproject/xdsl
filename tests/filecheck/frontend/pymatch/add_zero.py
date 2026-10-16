# RUN: python %s | filecheck %s

from xdsl.dialects import pdl, pdl_interp
from xdsl.dialects.builtin import IntegerAttr
from xdsl.frontend.pymatch import (
    check_operand_count,
    check_result_count,
    get_attribute,
    get_defining_op,
    get_operand,
    has_attr_value,
    has_name,
)
from xdsl.frontend.pymatch.context import PyMatchContext
from xdsl.ir import Operation, SSAValue

Captures = tuple[Operation, SSAValue]

rewriter = pdl_interp.FuncOp("rewrite", ([pdl.OperationType(), pdl.ValueType()], []))
target, replacement = rewriter.body.block.args
rewriter.body.block.add_ops(
    [pdl_interp.ReplaceOp(target, [replacement]), pdl_interp.FinalizeOp()]
)
ctx = PyMatchContext(rewriter)


@ctx.parse_program
def add_zero(root: Operation) -> Captures:
    """Match an i32 add whose right operand is zero; capture its left operand."""
    name = "arith.addi"
    has_name(name=name, op=root)
    check_operand_count(root, 2)
    check_result_count(root, 1)
    lhs = get_operand(root, 0)
    rhs = get_operand(root, index=1)
    constant = get_defining_op(rhs)
    has_name(constant, "arith.constant")
    value = get_attribute(constant, "value")
    has_attr_value(value, IntegerAttr(0, 32))
    captures = (root, lhs)
    return captures


print(add_zero.module)
assert rewriter.parent is None
# CHECK:      match.matcher root(%root: !pdl.operation) {
# CHECK-NEXT:   match.has_name %root, "arith.addi"
# CHECK-NEXT:   match.check_operand_count %root is 2
# CHECK-NEXT:   match.check_result_count %root is 1
# CHECK-NEXT:   %[[MAYBE_LHS:.*]] = match.get_operand 0 of %root : !match.optional<!pdl.value>
# CHECK-NEXT:   %[[LHS:.*]] = match.is_not_null %[[MAYBE_LHS]] : !pdl.value
# CHECK-NEXT:   %[[MAYBE_RHS:.*]] = match.get_operand 1 of %root : !match.optional<!pdl.value>
# CHECK-NEXT:   %[[RHS:.*]] = match.is_not_null %[[MAYBE_RHS]] : !pdl.value
# CHECK-NEXT:   %[[MAYBE_CONSTANT:.*]] = match.get_defining_op of %[[RHS]] : !pdl.value -> !match.optional<!pdl.operation>
# CHECK-NEXT:   %[[CONSTANT:.*]] = match.is_not_null %[[MAYBE_CONSTANT]] : !pdl.operation
# CHECK-NEXT:   match.has_name %[[CONSTANT]], "arith.constant"
# CHECK-NEXT:   %[[MAYBE_VALUE:.*]] = match.get_attribute "value" of %[[CONSTANT]] : !match.optional<!pdl.attribute>
# CHECK-NEXT:   %[[VALUE:.*]] = match.is_not_null %[[MAYBE_VALUE]] : !pdl.attribute
# CHECK-NEXT:   match.has_attr_value %[[VALUE]] is 0 : i32
# CHECK-NEXT:   match.success @rewrite benefit(1) (%root, %[[LHS]] : !pdl.operation, !pdl.value)
# CHECK-NEXT: }
# CHECK-NEXT: pdl_interp.func @rewrite(%{{.*}}: !pdl.operation, %{{.*}}: !pdl.value) {
# CHECK-NEXT:   pdl_interp.replace %{{.*}} with (%{{.*}} : !pdl.value)
# CHECK-NEXT:   pdl_interp.finalize
# CHECK-NEXT: }
