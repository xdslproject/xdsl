import pytest

from xdsl.ir import Block, Region
from xdsl.dialects.arith import Addi, Subi, Constant
from xdsl.dialects.builtin import i32, IntegerAttr


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


def test_ops_accessor_II():
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

    # Erase operations and block
    region2 = Region()
    region.move_blocks(region2)

    region2.blocks[0].erase_op(a, safe_erase=False)
    region2.blocks[0].erase_op(b, safe_erase=False)
    region2.blocks[0].erase_op(c, safe_erase=False)

    region2.detach_block(block0)
    region2.drop_all_references()

    assert len(region2.blocks) == 0


def test_ops_accessor_III():
    # Create constants `from_attr` and add them, add them in blocks, blocks in
    # a region and create a function
    a = Constant.from_attr(IntegerAttr.from_int_and_width(1, 32), i32)
    b = Constant.from_attr(IntegerAttr.from_int_and_width(2, 32), i32)
    c = Constant.from_attr(IntegerAttr.from_int_and_width(3, 32), i32)
    d = Constant.from_attr(IntegerAttr.from_int_and_width(4, 32), i32)

    # Operation to add these constants
    e = Addi.get(a, b)
    f = Addi.get(c, d)

    # Create Blocks and Regions
    block0 = Block.from_ops([a, b, e])
    block1 = Block.from_ops([c, d, f])
    region0 = Region.from_block_list([block0, block1])

    with pytest.raises(ValueError):
        region0.ops
        pass
