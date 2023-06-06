from xdsl.ir import Block, SSAValue
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.builtin import i32
from xdsl.utils.live_range import LiveRange


def test_live_range_no_use():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    c = Addi(a, b)

    _ = Block([a, b, c])

    lr0 = LiveRange(SSAValue.get(c))

    assert lr0.start == 2 and lr0.end == 2


def test_live_range_single_use():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    c = Addi(a, b)
    d = Addi(c, b)

    _ = Block([a, b, c, d])

    lr0 = LiveRange(SSAValue.get(c))

    assert lr0.start == 2 and lr0.end == 3


def test_live_range_multiple_uses():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    c = Addi(a, b)
    d = Addi(c, b)
    e = Addi(c, b)

    _ = Block([a, b, c, d, e])

    lr0 = LiveRange(SSAValue.get(c))

    assert lr0.start == 2 and lr0.end == 4
