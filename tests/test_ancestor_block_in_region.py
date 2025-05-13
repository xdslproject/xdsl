from xdsl.dialects import test
from xdsl.irdl import Block, Region


def test_ancestor_block_in_region_nested():
    # lvl 3
    op_3 = test.TestOp()

    # lvl 2
    blk_2 = Block([op_3])
    op_2 = test.TestOp(regions=[Region(blk_2)])

    # lvl 1
    blk_1 = Block([op_2])
    reg_1 = Region(blk_1)
    test.TestOp(regions=[reg_1])

    assert reg_1.find_ancestor_block_in_region(blk_2) is blk_1
    assert reg_1.find_ancestor_block_in_region(blk_1) is blk_1


def test_ancestor_block_in_region_different_region():
    # lvl 3
    op_3 = test.TestOp()

    # lvl 2
    blk_2 = Block([op_3])
    op_2 = test.TestOp(regions=[Region(blk_2)])

    # lvl 1
    blk_1 = Block([op_2])
    reg_1_1 = Region(blk_1)
    reg_1_2 = Region()
    test.TestOp(regions=[reg_1_1, reg_1_2])

    assert reg_1_2.find_ancestor_block_in_region(blk_2) is not blk_1
