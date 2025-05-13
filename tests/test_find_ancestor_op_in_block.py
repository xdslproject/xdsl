from xdsl.dialects import test
from xdsl.irdl import Block, Region


def test_find_ancestor_op_in_block_has_ancestor():
    op1 = test.TestOp()
    op2 = test.TestOp(regions=[Region(Block([op1]))])
    # test.TestOp(regions=[Region(Block([op2]))])

    blk_top = Block([op2])
    test.TestOp(regions=[Region(blk_top)])

    assert blk_top.find_ancestor_op_in_block(op1) is op2


def test_find_ancestor_op_in_block_has_no_ancestor():
    op1 = test.TestOp()
    op2 = test.TestOp(regions=[Region(Block([op1]))])
    # test.TestOp(regions=[Region(Block([op2]))])

    blk_top = Block([op2])
    op3 = test.TestOp(regions=[Region(blk_top)])

    assert blk_top.find_ancestor_op_in_block(op3) is None
