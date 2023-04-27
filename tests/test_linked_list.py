from xdsl.ir import Block
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import i32


def test_block_insert():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    c = Constant.from_int_and_width(3, i32)

    block = Block()

    assert block.is_empty
    
    assert block.first_op is None
    assert block.last_op is None

    assert list(block.ops) == []

    block.add_op(a)

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
