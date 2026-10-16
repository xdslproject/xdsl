from collections.abc import Callable
from unittest.mock import Mock

import pytest

from xdsl.context import Context
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
from xdsl.interpreter import Interpreter
from xdsl.interpreters.match import MatchResult, run_matcher
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    var_operand_def,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.convert_match_to_pdl_interp import ConvertMatchToPDLInterpPass


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
    for error in (ValueError("bad matcher"), MatchFailure()):
        match_result.side_effect = error
        with pytest.raises(type(error)) as exc:
            pattern.match_and_rewrite(root, rewriter)
        assert exc.value is error

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


@irdl_op_definition
class PairOp(IRDLOperation):
    """Deliberately unlike add-zero: renamed fields and multiple results."""

    name = "test.pymatch_pair"
    first = operand_def()
    second = operand_def()
    tail = var_operand_def()
    one = result_def()
    two = result_def()
    label = attr_def(builtin.StringAttr, attr_name="label-name")
    marker = prop_def(builtin.StringAttr, prop_name="marker-name")
    optional = opt_prop_def(
        builtin.StringAttr, default_value=builtin.StringAttr("default")
    )


def test_typed_fields_captures_and_rewrite():
    def matcher(op: Operation) -> tuple[Attribute, SSAValue, PairOp] | None:
        if not isinstance(op, PairOp):
            return None
        if op.label != builtin.StringAttr("selected"):
            return None
        if builtin.StringAttr("replace") != op.marker:
            return None
        return op.label, op.two, op

    @rewrite_for(matcher)
    def rewrite(
        label: Attribute, result: SSAValue, op: PairOp, rw: PatternRewriter
    ) -> None:
        rw.replace(op, new_ops=(), new_results=(op.second, op.first))

    module = rewrite.module
    assert rewrite.module is module
    matcher_op = module.body.block.first_op
    assert isinstance(matcher_op, match.MatcherOp)
    interpreter = Interpreter(module)
    interpreter.register_implementations(PDLInterpFunctions())
    block = Block(arg_types=[builtin.i32, builtin.i64])
    root = PairOp.build(
        operands=[block.args[0], block.args[1], []],
        result_types=[builtin.i64, builtin.i32],
        attributes={"label-name": builtin.StringAttr("selected")},
        properties={"marker-name": builtin.StringAttr("replace")},
    )
    block.add_op(root)
    root.verify()
    expected = (builtin.StringAttr("selected"), root.two, root)
    assert matcher(root) == expected
    assert run_matcher(matcher_op, root) == MatchResult(
        builtin.SymbolRefAttr("rewrite"), expected
    )
    rewriter = Mock(spec=PatternRewriter)
    rewrite(*expected, rewriter)
    rewriter.replace.assert_called_once_with(
        root, new_ops=(), new_results=(block.args[1], block.args[0])
    )
    rewriter.reset_mock()
    PDLInterpFunctions.set_rewriter(interpreter, rewriter)
    interpreter.call_op("rewrite", expected)
    rewriter.replace.assert_called_once_with(
        root, new_ops=[], new_results=[block.args[1], block.args[0]]
    )
    for label, marker in (("other", "replace"), ("selected", "other")):
        root.label = builtin.StringAttr(label)
        root.marker = builtin.StringAttr(marker)
        assert matcher(root) is None
        assert run_matcher(matcher_op, root) is None
    assert matcher(test.TestOp()) is None
    assert run_matcher(matcher_op, test.TestOp()) is None


def test_unused_owner_and_empty_captures():
    def matcher(op: PairOp) -> tuple[()] | None:
        _owner = op.second.owner
        return ()

    @rewrite_for(matcher)
    def rewrite(rw: PatternRewriter) -> None:
        pass

    block = Block(arg_types=[builtin.i32, builtin.i64])
    root = PairOp.build(
        operands=[block.args[0], block.args[1], []],
        result_types=[builtin.i64, builtin.i32],
        attributes={"label-name": builtin.StringAttr("selected")},
        properties={"marker-name": builtin.StringAttr("replace")},
    )
    block.add_op(root)
    root.verify()
    assert matcher(root) == ()
    module = rewrite.module
    matcher_op = module.body.block.first_op
    assert isinstance(matcher_op, match.MatcherOp)
    assert run_matcher(matcher_op, root) == MatchResult(
        builtin.SymbolRefAttr("rewrite"), ()
    )
    lowered = module.clone()
    ConvertMatchToPDLInterpPass().apply(Context(), lowered)
    lowered.verify()
    interpreter = Interpreter(lowered)
    interpreter.register_implementations(PDLInterpFunctions())
    PDLInterpFunctions.set_rewriter(interpreter, PatternRewriter(root))
    interpreter.call_op("matcher", (root,))
    assert PDLInterpFunctions.get_pending_rewrites(interpreter) == [
        (builtin.SymbolRefAttr("rewrite"), root, ())
    ]


def test_typed_frontend_rejects_unsupported_syntax():
    def success_branch(op: PairOp) -> tuple[PairOp] | None:
        if op.label != builtin.StringAttr("other"):
            return (op,)
        return None

    def loop(op: PairOp) -> tuple[PairOp] | None:
        for _ in op.operands:
            pass
        return (op,)

    def variadic(op: PairOp) -> tuple[PairOp] | None:
        _values = op.tail
        return (op,)

    def optional(op: PairOp) -> tuple[PairOp] | None:
        _attribute = op.optional
        return (op,)

    def rewrite(op: PairOp, rw: PatternRewriter) -> None:
        pass

    for matcher, message in (
        (success_branch, "failure guard"),
        (loop, "unsupported statement: For"),
        (variadic, "optional/variadic fields"),
        (optional, "optional/defaulted attribute"),
    ):
        pattern = rewrite_for(matcher)(rewrite)
        with pytest.raises(FrontendProgramException, match=message):
            pattern.module

    def identity(op: PairOp) -> tuple[PairOp] | None:
        return (op,)

    @rewrite_for(identity)
    def unsupported_rewrite(op: PairOp, rw: PatternRewriter) -> None:
        rw.replace(op, (), (op.first, op.second), safe_erase=False)

    with pytest.raises(FrontendProgramException, match="replacement requires"):
        unsupported_rewrite.module
