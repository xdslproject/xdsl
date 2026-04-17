import pytest

from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    IntegerAttr,
    Signedness,
)
from xdsl.traits import ConstantLike
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import VerifyException


def test_immediate_pseudo_inst():
    lb, ub = Signedness.SIGNLESS.value_range(32)
    assert ub == 4294967296
    assert lb == -2147483648

    # Pseudo-Instruction with custom handling
    with pytest.raises(VerifyException):
        rv32.LiOp(ub, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.LiOp(lb - 1, rd=riscv.Registers.A0)

    rv32.LiOp(ub - 1, rd=riscv.Registers.A0)
    rv32.LiOp(lb, rd=riscv.Registers.A0)


def test_get_constant_value():
    # Test 32-bit LiOp
    li_op = rv32.LiOp(1)
    li_val = get_constant_value(li_op.rd)
    assert li_val == IntegerAttr(1, 32)
    # LiOp implements ConstantLikeInterface so it also has a get_constant_value method:
    constantlike = li_op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(li_op.rd) == IntegerAttr(1, 32)

    # Test 32-bit zero register
    zero_op = rv32.GetRegisterOp(riscv.Registers.ZERO)
    zero_val = get_constant_value(zero_op.res)
    assert zero_val == IntegerAttr(0, 32)
