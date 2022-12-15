from xdsl.ir import Block, Region
from xdsl.dialects.arith import Addi, Subi, Constant
from xdsl.dialects.builtin import i32


def test_ops_accessor():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block.from_ops([a, b, c])
    # Create a region to include a, b, c
    region = Region.from_block_list([block0])

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi.get(a, b)

    assert d.results[0] != c.results[0]
