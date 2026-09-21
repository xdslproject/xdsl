import io

import pytest

from xdsl.context import Context
from xdsl.dialects import test, transform
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, f32, i32, i64
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_match_interface_enum_matches_mlir_values():
    """
    The position of each case is the `i32` value MLIR stores for it.
    """
    assert transform.MATCH_INTERFACES == (
        transform.MatchInterfaceEnum.LINALG_OP,
        transform.MatchInterfaceEnum.TILING_INTERFACE,
        transform.MatchInterfaceEnum.LOOP_LIKE_INTERFACE,
    )


@pytest.mark.parametrize(
    "interface, expected",
    [
        (transform.MatchInterfaceEnum.TILING_INTERFACE, IntegerAttr(1, i32)),
        (2, IntegerAttr(2, i32)),
        (IntegerAttr(0, i32), IntegerAttr(0, i32)),
        (None, None),
    ],
)
def test_match_op_interface_constructor(
    interface: transform.MatchInterfaceEnum | int | IntegerAttr | None,
    expected: IntegerAttr | None,
):
    target = create_ssa_value(transform.AnyOpType())
    op = transform.MatchOp(target, interface=interface)
    op.verify()
    assert op.interface == expected


def test_match_op_constructor_converts_sequences():
    target = create_ssa_value(transform.AnyOpType())
    op = transform.MatchOp(
        target,
        ops=["linalg.matmul", "linalg.generic"],
        op_attrs={"foo": IntegerAttr(1, i64)},
        filter_result_type=f32,
        filter_operand_types=[f32, f32],
    )
    op.verify()
    assert op.ops == ArrayAttr(
        [transform.StringAttr("linalg.matmul"), transform.StringAttr("linalg.generic")]
    )
    assert op.filter_operand_types == ArrayAttr([f32, f32])


def test_match_op_rejects_out_of_range_interface():
    target = create_ssa_value(transform.AnyOpType())
    op = transform.MatchOp(target, interface=3)
    with pytest.raises(VerifyException, match="interface must be one of"):
        op.verify()


def test_match_op_custom_syntax_roundtrip():
    ctx = Context()
    ctx.load_dialect(test.Test)
    ctx.load_dialect(transform.Transform)

    text = (
        '%0 = "test.op"() : () -> !transform.any_op\n'
        '%1 = transform.structured.match ops{["linalg.matmul"]} interface{TilingInterface} '
        "attributes {foo = 1 : i64} filter_result_type = f32 "
        "filter_operand_types = [f32, f32] in %0 : (!transform.any_op) -> !transform.any_op"
    )
    module = Parser(ctx, text).parse_module()
    op = module.ops.last
    assert isinstance(op, transform.MatchOp)
    assert op.interface == IntegerAttr(1, i32)

    out = io.StringIO()
    Printer(stream=out).print_op(module)
    assert out.getvalue() == "builtin.module {\n  " + text.replace("\n", "\n  ") + "\n}"


def test_match_op_rejects_unknown_interface_keyword():
    ctx = Context()
    ctx.load_dialect(test.Test)
    ctx.load_dialect(transform.Transform)
    with pytest.raises(Exception, match="expected one of LinalgOp"):
        Parser(
            ctx,
            '%0 = "test.op"() : () -> !transform.any_op\n'
            "%1 = transform.structured.match interface{Nope} in %0 : (!transform.any_op) -> !transform.any_op",
        ).parse_module()
