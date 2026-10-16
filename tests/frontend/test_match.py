from collections.abc import Callable
from unittest.mock import Mock

import pytest

from xdsl.dialects import builtin, match, pdl, pdl_interp, test
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException
from xdsl.frontend.pymatch import (
    MatchFailure,
    PythonMatchRewritePattern,
    check_operand_count,
    check_result_count,
    get_attribute,
    get_defining_op,
    get_operand,
    get_result,
    has_attr_value,
    has_name,
    rewrite_for,
)
from xdsl.frontend.pymatch.context import PyMatchContext
from xdsl.interpreters.match import MatchResult, run_matcher
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def test_scalar_helpers():
    block = Block(arg_types=[builtin.i32, builtin.i64])
    root = test.TestOp(
        block.args,
        [builtin.i64],
        {"a": builtin.StringAttr("attribute")},
        {"p": builtin.StringAttr("property")},
    )
    has_name(root, "test.op")
    check_operand_count(root, 2)
    check_result_count(root, 1)
    assert get_operand(root, 1) is block.args[1]
    assert get_result(root, 0) is root.results[0]
    assert get_defining_op(root.results[0]) is root
    assert get_attribute(root, "p") == builtin.StringAttr("property")
    has_attr_value(get_attribute(root, "a"), builtin.StringAttr("attribute"))

    failures: list[tuple[Callable[..., object], tuple[object, ...]]] = [
        (has_name, (root, "other")),
        (check_operand_count, (root, 1)),
        (check_result_count, (root, 2)),
        (get_operand, (root, 2)),
        (get_result, (root, 1)),
        (get_defining_op, (block.args[0],)),
        (get_attribute, (root, "missing")),
        (has_attr_value, (builtin.IntegerAttr(0, 32), builtin.IntegerAttr(0, 64))),
    ]
    for helper, args in failures:
        with pytest.raises(MatchFailure):
            helper(*args)

    for helper in (get_operand, get_result, check_operand_count, check_result_count):
        with pytest.raises(ValueError, match="nonnegative"):
            helper(root, -1)


def test_python_rewrite_captures_and_failures():
    root = test.TestOp(result_types=[builtin.i32])
    Block([root])
    rewriter = PatternRewriter(root)
    captures = (root, root.results[0])
    match_result = Mock(return_value=captures)

    def matcher(op: test.TestOp) -> tuple[test.TestOp, SSAValue] | None:
        return match_result(op)

    rewrite = Mock()
    pattern = rewrite_for(matcher)(rewrite)
    pattern.match_and_rewrite(builtin.ModuleOp([]), rewriter)
    match_result.assert_not_called()
    pattern.match_and_rewrite(root, rewriter)
    match_result.assert_called_once_with(root)
    rewrite.assert_called_once_with(*captures, rewriter)
    rewrite.reset_mock()

    match_result.return_value = None
    pattern.match_and_rewrite(root, rewriter)
    rewrite.assert_not_called()
    match_result.side_effect = ValueError("bad matcher")
    with pytest.raises(ValueError, match="bad matcher"):
        pattern.match_and_rewrite(root, rewriter)

    match_result.side_effect = None
    match_result.return_value = captures
    rewrite.side_effect = MatchFailure()
    with pytest.raises(MatchFailure):
        pattern.match_and_rewrite(root, rewriter)


def test_empty_captures_are_successful():
    def matcher(op: Operation) -> tuple[()] | None:
        return ()

    rewrite = Mock()
    pattern = PythonMatchRewritePattern(matcher, rewrite)
    root = test.TestOp()
    Block([root])
    rewriter = PatternRewriter(root)
    pattern.match_and_rewrite(root, rewriter)
    rewrite.assert_called_once_with(rewriter)
    rewrite.reset_mock()
    pattern(rewriter)
    rewrite.assert_called_once_with(rewriter)


def test_rewrite_capture_typing():
    """Pyright also checks these contracts, including the expected diagnostics."""

    def matcher(op: test.TestOp) -> tuple[test.TestOp, SSAValue] | None:
        return op, op.results[0]

    def compatible(op: Operation, value: SSAValue, rewriter: PatternRewriter) -> None:
        pass

    def swapped(value: SSAValue, op: test.TestOp, rewriter: PatternRewriter) -> None:
        pass

    def missing(op: test.TestOp, rewriter: PatternRewriter) -> None:
        pass

    def wrong_type(
        op: builtin.ModuleOp, value: SSAValue, rewriter: PatternRewriter
    ) -> None:
        pass

    rewrite_for(matcher)(compatible)
    rewrite_for(matcher)(swapped)  # pyright: ignore[reportArgumentType]
    rewrite_for(matcher)(missing)  # pyright: ignore[reportArgumentType]
    rewrite_for(matcher)(wrong_type)  # pyright: ignore[reportArgumentType]


def test_typed_capture_order_and_discarded_navigation():
    rewriter = pdl_interp.FuncOp(
        "rewrite", ([pdl.AttributeType(), pdl.ValueType(), pdl.OperationType()], [])
    )
    rewriter.body.block.add_op(pdl_interp.FinalizeOp())
    ctx = PyMatchContext(rewriter)

    @ctx.parse_program
    def captures(root: Operation) -> tuple[Attribute, SSAValue, Operation]:
        get_operand(root, 1)
        attribute = get_attribute(root, "p")
        has_attr_value(attribute, builtin.IntegerAttr(7, 64))
        return attribute, get_result(root, 1), root

    module = captures.module
    matcher = module.body.block.first_op
    assert isinstance(matcher, match.MatcherOp)
    block = Block(arg_types=[builtin.i32, builtin.i64])
    root = test.TestOp(
        block.args,
        [builtin.i32, builtin.i64],
        properties={"p": builtin.IntegerAttr(7, 64)},
    )
    expected = (builtin.IntegerAttr(7, 64), root.results[1], root)
    assert captures(root) == expected
    assert run_matcher(matcher, root) == MatchResult(
        builtin.SymbolRefAttr("rewrite"), expected
    )
    root.operands = [block.args[0]]
    with pytest.raises(MatchFailure):
        captures(root)
    assert run_matcher(matcher, root) is None
    root.operands = block.args
    for wrong_attribute in (builtin.IntegerAttr(0, 64), builtin.IntegerAttr(7, 32)):
        root.properties["p"] = wrong_attribute
        with pytest.raises(MatchFailure):
            captures(root)
        assert run_matcher(matcher, root) is None


def test_frontend_rejects_unsupported_forms():
    rewriter = pdl_interp.FuncOp("rewrite", ([pdl.OperationType()], []))
    rewriter.body.block.add_op(pdl_interp.FinalizeOp())
    ctx = PyMatchContext(rewriter)

    with pytest.raises(FrontendProgramException, match="straight-line"):

        @ctx.parse_program
        def _conditional(root: Operation) -> tuple[Operation]:
            if root.name == "test.op":
                return (root,)
            return (root,)

    # PyAST does not yet implement annotated assignments; do not silently omit checks.
    with pytest.raises(FrontendProgramException, match="straight-line"):

        @ctx.parse_program
        def _annotated(root: Operation) -> tuple[Operation]:
            _value: SSAValue = get_operand(root, 1)
            return (root,)

    with pytest.raises(FrontendProgramException, match="fixed, flat tuple"):

        @ctx.parse_program
        def _variadic(root: Operation) -> tuple[Operation, ...]:
            return (root,)

    @ctx.parse_program
    def wrong_name_type(root: Operation) -> tuple[Operation]:
        has_name(root, 7)  # pyright: ignore[reportArgumentType]
        return (root,)

    @ctx.parse_program
    def wrong_attribute_type(root: Operation) -> tuple[Operation]:
        has_attr_value(get_attribute(root, "p"), 0)  # pyright: ignore[reportArgumentType]
        return (root,)

    @ctx.parse_program
    def dynamic_constant(root: Operation) -> tuple[Operation]:
        has_attr_value(get_attribute(root, "p"), get_attribute(root, "q"))
        return (root,)

    @ctx.parse_program
    def wrong_capture_type(root: Operation) -> tuple[SSAValue]:
        return (get_result(root, 0),)

    for program, message in (
        (wrong_name_type, "constant Python string"),
        (wrong_attribute_type, "constructed with IntegerAttr"),
        (dynamic_constant, "compile-time constant"),
        (wrong_capture_type, "capture types do not match"),
    ):
        with pytest.raises(FrontendProgramException, match=message):
            program.module
