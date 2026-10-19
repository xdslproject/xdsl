import pytest

from xdsl.dialects import transform
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, f32, i32, i64
from xdsl.utils.test_value import create_ssa_value


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
        [StringAttr("linalg.matmul"), StringAttr("linalg.generic")]
    )
    assert op.filter_operand_types == ArrayAttr([f32, f32])
