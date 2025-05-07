import pytest

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import i32
from xdsl.ir import Block, Region


def test_block_insert():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    c = ConstantOp.from_int_and_width(3, i32)
    d = ConstantOp.from_int_and_width(4, i32)
    e = ConstantOp.from_int_and_width(5, i32)

    block = Block()

    assert not block.ops

    assert block.first_op is None
    assert block.last_op is None

    assert list(block.ops) == []

    block.add_op(a)

    assert block.ops

    assert block.first_op is a
    assert block.last_op is a

    assert list(block.ops) == [a]

    block.add_op(c)

    assert block.first_op is a
    assert block.last_op is c

    assert a.next_op is c
    assert c.prev_op is a

    assert list(block.ops) == [a, c]

    block.insert_op_after(b, a)

    assert block.first_op is a
    assert block.last_op is c

    assert a.next_op is b
    assert b.prev_op is a

    assert b.next_op is c
    assert c.prev_op is b

    assert list(block.ops) == [a, b, c]

    block.add_op(e)

    assert block.first_op is a
    assert block.last_op is e

    assert c.next_op is e
    assert e.prev_op is c

    assert list(block.ops) == [a, b, c, e]

    block.insert_op_before(d, e)

    assert block.first_op is a
    assert block.last_op is e

    assert c.next_op is d
    assert d.prev_op is c

    assert d.next_op is e
    assert e.prev_op is d

    assert list(block.ops) == [a, b, c, d, e]


def test_region_insert():
    a = Block((ConstantOp.from_int_and_width(1, i32),))
    b = Block((ConstantOp.from_int_and_width(2, i32),))
    c = Block((ConstantOp.from_int_and_width(3, i32),))
    d = Block((ConstantOp.from_int_and_width(4, i32),))
    e = Block((ConstantOp.from_int_and_width(5, i32),))

    region = Region()

    assert not region.blocks

    assert region.first_block is None
    assert region.last_block is None

    assert list(region.blocks) == []

    region.add_block(a)

    assert region.blocks

    assert region.first_block is a
    assert region.last_block is a

    assert list(region.blocks) == [a]

    region.add_block(c)

    assert region.first_block is a
    assert region.last_block is c

    assert a.next_block is c
    assert c.prev_block is a

    assert list(region.blocks) == [a, c]

    region.insert_block_after(b, a)

    assert region.first_block is a
    assert region.last_block is c

    assert a.next_block is b
    assert b.prev_block is a

    assert b.next_block is c
    assert c.prev_block is b

    assert list(region.blocks) == [a, b, c]

    region.add_block(e)

    assert region.first_block is a
    assert region.last_block is e

    assert c.next_block is e
    assert e.prev_block is c

    assert list(region.blocks) == [a, b, c, e]

    region.insert_block_before(d, e)

    assert region.first_block is a
    assert region.last_block is e

    assert c.next_block is d
    assert d.prev_block is c

    assert d.next_block is e
    assert e.prev_block is d

    assert list(region.blocks) == [a, b, c, d, e]


def test_region_indexing():
    a = Block((ConstantOp.from_int_and_width(1, i32),))
    b = Block((ConstantOp.from_int_and_width(2, i32),))
    c = Block((ConstantOp.from_int_and_width(3, i32),))
    d = Block((ConstantOp.from_int_and_width(4, i32),))
    e = Block((ConstantOp.from_int_and_width(5, i32),))

    region = Region((a, b, c, d, e))

    assert region.blocks[0] == a
    assert region.blocks[1] == b
    assert region.blocks[2] == c
    assert region.blocks[3] == d
    assert region.blocks[4] == e

    with pytest.raises(IndexError):
        region.blocks[5]

    assert region.blocks[-1] == e
    assert region.blocks[-2] == d
    assert region.blocks[-3] == c
    assert region.blocks[-4] == b
    assert region.blocks[-5] == a

    with pytest.raises(IndexError):
        region.blocks[-6]
