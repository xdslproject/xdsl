import pytest

from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    IntegerAttr,
    Signedness,
)
from xdsl.traits import ConstantLike
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


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


def test_immediate_shift_inst():
    # Shift instructions (SLLI, SRLI, SRAI) - 5-bits immediate
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.SlliOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.SlliOp(a1, -1, rd=riscv.Registers.A0)

    rv32.SlliOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_get_constant_value():
    li_op = rv32.LiOp(1)
    li_val = get_constant_value(li_op.rd)
    assert li_val == IntegerAttr.from_int_and_width(1, 32)
    # LiOp implements ConstantLikeInterface so it also has a get_constant_value method:
    constantlike = li_op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(li_op) == IntegerAttr.from_int_and_width(
        1, 32
    )
    zero_op = riscv.GetRegisterOp(riscv.Registers.ZERO)
    zero_val = get_constant_value(zero_op.res)
    assert zero_val == IntegerAttr.from_int_and_width(0, 32)
