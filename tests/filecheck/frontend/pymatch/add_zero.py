# RUN: python %s | filecheck %s

from pathlib import Path
from unittest.mock import patch

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, match, pdl, pdl_interp
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, SymbolRefAttr, i32, i64
from xdsl.frontend.pymatch import (
    MatchFailure,
    PythonMatchRewritePattern,
    check_operand_count,
    check_result_count,
    get_attribute,
    get_defining_op,
    get_operand,
    has_attr_value,
    has_name,
)
from xdsl.frontend.pymatch.context import PyMatchContext
from xdsl.interpreter import Interpreter
from xdsl.interpreters.match import MatchResult, run_matcher
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker
from xdsl.transforms.apply_match import ApplyMatchPass
from xdsl.transforms.apply_pdl_interp import ApplyPDLInterpPass
from xdsl.transforms.convert_match_to_pdl_interp import ConvertMatchToPDLInterpPass

Captures = tuple[Operation, SSAValue]

rewriter = pdl_interp.FuncOp(
    "rewrite_add_zero", ([pdl.OperationType(), pdl.ValueType()], [])
)
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
# CHECK-NEXT:   match.success @rewrite_add_zero benefit(1) (%root, %[[LHS]] : !pdl.operation, !pdl.value)
# CHECK-NEXT: }
# CHECK-NEXT: pdl_interp.func @rewrite_add_zero(%{{.*}}: !pdl.operation, %{{.*}}: !pdl.value) {
# CHECK-NEXT:   pdl_interp.replace %{{.*}} with (%{{.*}} : !pdl.value)
# CHECK-NEXT:   pdl_interp.finalize
# CHECK-NEXT: }


# The same captures must reach the rewrite in all three execution paths.
matcher = add_zero.module.body.block.first_op
assert isinstance(matcher, match.MatcherOp)
lowered = add_zero.module.clone()
ConvertMatchToPDLInterpPass().apply(Context(), lowered)
lowered.verify()
lowered_interpreter = Interpreter(lowered)
pdl_functions = PDLInterpFunctions()
lowered_interpreter.register_implementations(pdl_functions)

block = Block(arg_types=[i32, i64])
x, y = block.args
zero = arith.ConstantOp.from_int_and_width(0, i32)
seven = arith.ConstantOp.from_int_and_width(7, i32)
wide_zero = arith.ConstantOp.from_int_and_width(0, i64)
first = arith.AddiOp(x, zero)
second = arith.AddiOp(seven, zero)
cases = (
    (first, (first, x)),
    (arith.AddiOp(x, seven), None),
    (second, (second, seven.result)),
    (arith.AddiOp(zero, x), None),
    (arith.SubiOp(x, zero), None),
    (arith.AddiOp(y, wide_zero), None),
)
block.add_ops([zero, seven, wide_zero, *(root for root, _ in cases), func.ReturnOp()])
payload = func.FuncOp("captures", ([i32, i64], []), Region(block))
payload.verify()
before = payload.clone()
for root, expected in cases:
    try:
        python_captures = add_zero(root)
    except MatchFailure:
        python_captures = None
    assert python_captures == expected
    assert run_matcher(matcher, root) == (
        None
        if expected is None
        else MatchResult(SymbolRefAttr("rewrite_add_zero"), expected)
    )
    pdl_functions.set_rewriter(lowered_interpreter, PatternRewriter(root))
    lowered_interpreter.call_op("matcher", (root,))
    with patch.object(lowered_interpreter, "call_op") as rewrite:
        pdl_functions.apply_pending_rewrites(lowered_interpreter)
        if expected is None:
            rewrite.assert_not_called()
        else:
            rewrite.assert_called_once_with(SymbolRefAttr("rewrite_add_zero"), expected)
assert payload.is_structurally_equivalent(before)


def rewrite_add_zero(captures: Captures, rewriter: PatternRewriter) -> None:
    root, lhs = captures
    rewriter.replace(root, [], [lhs])


# Reuse the real payload fixture, including dependent rewrites and nonmatches.
ir_ctx = Context()
for dialect in (builtin.Builtin, arith.Arith, func.Func):
    ir_ctx.load_dialect(dialect)
path = (
    Path(__file__).parents[2] / "transforms/convert-match-to-pdl-interp/add-zero.mlir"
)
source = Parser(ir_ctx, path.read_text()).parse_module()
source.verify()
source_func = source.body.block.first_op
assert isinstance(source_func, func.FuncOp)

python_result = source_func.clone()
python_module = ModuleOp([python_result])
PatternRewriteWalker(
    PythonMatchRewritePattern(add_zero, rewrite_add_zero)
).rewrite_module(python_module)
python_module.verify()

for via_pdl_interp in (False, True):
    compiled_module = add_zero.module.clone()
    result = source_func.clone()
    compiled_module.body.block.add_op(result)
    compiled_module.verify()
    if via_pdl_interp:
        ConvertMatchToPDLInterpPass().apply(ir_ctx, compiled_module)
        ApplyPDLInterpPass().apply(ir_ctx, compiled_module)
    else:
        ApplyMatchPass().apply(ir_ctx, compiled_module)
    compiled_module.verify()
    # Compare detached payloads, not their different enclosing pattern modules.
    assert result.clone().is_structurally_equivalent(python_result.clone())

# Check the common output independently of the implementations being compared.
print(python_result)
# CHECK:      func.func @rewrite(%x: i32) -> (i32, i32, i32, i32, i32) {
# CHECK-NEXT:   %zero = arith.constant 0 : i32
# CHECK-NEXT:   %seven = arith.constant 7 : i32
# CHECK-NEXT:   %nonzero = arith.addi %x, %seven : i32
# CHECK-NEXT:   %block_arg = arith.addi %zero, %x : i32
# CHECK-NEXT:   %wrong_name = arith.subi %x, %zero : i32
# CHECK-NEXT:   func.return %x, %nonzero, %block_arg, %wrong_name, %nonzero : i32, i32, i32, i32, i32
# CHECK-NEXT: }
