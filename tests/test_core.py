from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.dialects.vector import PrintOp
from xdsl.ir import Block


def test_permute_block_ops():
    a = ConstantOp(IntegerAttr.from_int_and_width(1, 32), i32)
    b = ConstantOp(IntegerAttr.from_int_and_width(2, 32), i32)

    # Operations on these constants
    c = AddiOp(a, b)
    d = AddiOp(a, b)
    e = AddiOp(c, d)
    f = PrintOp.get(e)

    # Create Block from operations
    block0 = Block([a, b, c, d, e, f])
    orderings = [2, 5, 0, 4, 1, 3]
    block0.permute(orderings)

    expected_order = [c, f, a, e, b, d]
    expected_iter = iter(expected_order)
    for actual_op in block0.ops:
        expected_op = next(expected_iter)
        assert actual_op is expected_op
