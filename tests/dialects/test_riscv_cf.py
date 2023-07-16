import pytest

from xdsl.dialects import riscv, riscv_cf
from xdsl.ir.core import Block, Region
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_branch_validation():
    lhs = TestSSAValue(riscv.RegisterType(riscv.Register()))
    rhs = TestSSAValue(riscv.RegisterType(riscv.Register()))
    region = Region((Block(), Block(), Block()))
    first_block, then_block, else_block = region.blocks
    beq = riscv_cf.BeqOp(lhs, rhs, (), (), then_block, else_block)
    first_block.add_op(beq)

    with pytest.raises(
        VerifyException, match="riscv_cf branch op then block must not be empty"
    ):
        region.verify()

    comment = riscv.CommentOp("comment")
    then_block.add_op(comment)

    with pytest.raises(
        VerifyException, match="riscv_cf branch op then block first op must be a label"
    ):
        region.verify()

    then_block.insert_op_before(riscv.LabelOp("label"), comment)

    with pytest.raises(
        VerifyException,
        match="riscv_cf branch op else block must be immediately after op",
    ):
        region.verify()
