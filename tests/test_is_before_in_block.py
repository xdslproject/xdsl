from xdsl.dialects import test
from xdsl.irdl import Block, Region


def test_is_before_in_block_correct():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1, op2]))])

    assert op1.is_before_in_block(op2)


def test_is_before_in_block_incorrect():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1, op2]))])

    assert not op2.is_before_in_block(op1)


def test_is_before_in_block_same_op():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1, op2]))])

    assert not op1.is_before_in_block(op1)
