from xdsl.dialects import test
from xdsl.irdl import Block, Region


def test_same_block():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1, op2]))])

    assert op1.is_before_in_block(op2)
    assert not op2.is_before_in_block(op1)
    assert not op1.is_before_in_block(op1)


def test_different_blocks():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1])), Region(Block([op2]))])

    assert not op1.is_before_in_block(op2)
    assert not op2.is_before_in_block(op1)
