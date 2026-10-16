from pathlib import Path
from unittest.mock import patch

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, match, pdl, pdl_interp, test
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.match import MatchResult, run_match_op, run_matcher
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.convert_match_to_pdl_interp import ConvertMatchToPDLInterpPass
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


def test_navigation():
    lhs, rhs = create_ssa_value(i32), create_ssa_value(i64)
    root = test.TestOp(
        [lhs, rhs],
        [i32, i64],
        {"a": StringAttr("attribute")},
        {"p": StringAttr("property")},
    )
    handle = create_ssa_value(pdl.OperationType())
    for op_type, values in (
        (match.GetOperandOp, root.operands),
        (match.GetResultOp, root.results),
    ):
        assert run_match_op(op_type(1, handle), (root,)) == (values[1],)
        assert run_match_op(op_type(2, handle), (root,)) == (None,)
        with pytest.raises(InterpretationError, match="nonnegative index"):
            run_match_op(op_type(-1, handle), (root,))
        with pytest.raises(InterpretationError, match="optional value result"):
            run_match_op(
                op_type(0, handle, match.OptionalType(pdl.TypeType())), (root,)
            )

    for name, expected in (
        ("a", StringAttr("attribute")),
        ("p", StringAttr("property")),
        ("missing", None),
    ):
        assert run_match_op(match.GetAttributeOp(name, handle), (root,)) == (expected,)

    defining = match.GetDefiningOpOp(create_ssa_value(pdl.ValueType()))
    assert run_match_op(defining, (root.results[1],)) == (root,)
    assert run_match_op(defining, (Block(arg_types=[i32]).args[0],)) == (None,)
    unwrap = match.IsNotNullOp(create_ssa_value(match.OptionalType(pdl.ValueType())))
    assert run_match_op(unwrap, (rhs,)) == (rhs,)
    assert run_match_op(unwrap, (None,)) is None


@pytest.mark.parametrize(
    "op_type,count", [(match.CheckOperandCountOp, 2), (match.CheckResultCountOp, 3)]
)
@pytest.mark.parametrize(
    "delta,at_least,passes",
    [
        (-1, False, False),
        (0, False, True),
        (1, False, False),
        (-1, True, True),
        (0, True, True),
        (1, True, False),
    ],
)
def test_counts(
    op_type: type[match.CheckOperandCountOp | match.CheckResultCountOp],
    count: int,
    delta: int,
    at_least: bool,
    passes: bool,
):
    root = test.TestOp([create_ssa_value(i32), create_ssa_value(i64)], [i32, i64, i32])
    op = op_type(create_ssa_value(pdl.OperationType()), count + delta, at_least)
    assert run_match_op(op, (root,)) == (() if passes else None)


def test_attribute_equality():
    op = match.HasAttrValueOp(
        create_ssa_value(pdl.AttributeType()), IntegerAttr(0, i32)
    )
    assert run_match_op(op, (IntegerAttr(0, i32),)) == ()
    for value in (IntegerAttr(1, i32), IntegerAttr(0, i64)):
        assert run_match_op(op, (value,)) is None


def test_add_zero_captures_and_reuse():
    ctx = Context()
    for dialect in (builtin.Builtin, match.Match, pdl.PDL, pdl_interp.PDLInterp):
        ctx.load_dialect(dialect)
    path = Path(__file__).parent.parent / "filecheck/dialects/match/add-zero.mlir"
    module = Parser(ctx, path.read_text()).parse_module()
    module.verify()
    matcher = next(op for op in module.walk() if isinstance(op, match.MatcherOp))
    lowered = module.clone()
    ConvertMatchToPDLInterpPass().apply(ctx, lowered)
    lowered.verify()
    lowered_interpreter = Interpreter(lowered)
    pdl_functions = PDLInterpFunctions()
    lowered_interpreter.register_implementations(pdl_functions)

    block = Block(arg_types=[i32])
    lhs = block.args[0]
    zero = arith.ConstantOp.from_int_and_width(0, i32)
    seven = arith.ConstantOp.from_int_and_width(7, i32)
    first = arith.AddiOp(lhs, zero)
    second = arith.AddiOp(seven, zero)
    cases = (
        (first, (first, lhs)),
        (arith.AddiOp(lhs, seven), None),
        (second, (second, seven.result)),
        (arith.AddiOp(zero, lhs), None),
        (arith.SubiOp(lhs, zero), None),
    )
    block.add_ops(
        [zero, seven, *(root for root, _ in cases), func.ReturnOp(first, second)]
    )
    payload = ModuleOp([func.FuncOp("payload", ([i32], [i32, i32]), Region(block))])
    payload.verify()
    before = payload.clone()
    symbol = SymbolRefAttr("rewriters", ["add_zero"])
    for root, expected in cases:
        assert run_matcher(matcher, root) == (
            None if expected is None else MatchResult(symbol, expected)
        )
        pdl_functions.set_rewriter(lowered_interpreter, PatternRewriter(root))
        lowered_interpreter.call_op("matcher", (root,))
        with patch.object(lowered_interpreter, "call_op") as rewrite:
            pdl_functions.apply_pending_rewrites(lowered_interpreter)
            if expected is None:
                rewrite.assert_not_called()
            else:
                rewrite.assert_called_once_with(symbol, expected)
    assert payload.is_structurally_equivalent(before)


def test_failure_stops_before_using_results():
    matcher = match.MatcherOp()
    operand = match.GetOperandOp(0, matcher.body.block.args[0])
    unwrap = match.IsNotNullOp(operand.result)
    matcher.body.block.add_ops(
        [
            operand,
            unwrap,
            test.TestOp([unwrap.unwrapped]),
            match.SuccessOp("rewrite", 1),
        ]
    )
    assert run_matcher(matcher, test.TestOp()) is None
    # On success the unsupported consumer is reached, and must raise rather than
    # turn a programming error into an ordinary mismatch.
    with pytest.raises(
        InterpretationError, match=r"unsupported Match operation: test\.op"
    ):
        run_matcher(matcher, test.TestOp([create_ssa_value(i32)]))


def test_invalid_matcher_is_not_a_mismatch():
    matcher = match.MatcherOp()
    matcher.body.block.add_ops([test.TestOp(), match.SuccessOp("rewrite", 1)])
    with pytest.raises(
        InterpretationError, match=r"unsupported Match operation: test\.op"
    ):
        run_matcher(matcher, test.TestOp())

    matcher.body.block.add_op(match.SuccessOp("rewrite", 1))
    with pytest.raises(InterpretationError, match=r"expected one match\.success"):
        run_matcher(matcher, test.TestOp())
