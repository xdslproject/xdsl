import pytest

from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    IntegerAttr,
    Signedness,
    i32,
)
from xdsl.traits import ConstantLike, MemoryEffect, NoMemoryEffect, Pure
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


@pytest.mark.parametrize(
    "op_type",
    [rv32.BclrIOp, rv32.BextIOp, rv32.BsetIOp, rv32.BinvIOp, rv32.RorIOp],
)
def test_immediate_bit_manipulation_inst(
    op_type: type[rv32.RV32RdRsImmShiftOperation],
):
    # Bit manipulation instructions - 5-bits immediate,
    # shamt[5]=1 encodings are reserved on RV32
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        op_type(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        op_type(a1, -1, rd=riscv.Registers.A0)

    op_type(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


@pytest.mark.parametrize(
    ("op_type", "shamt", "rs1", "expected"),
    [
        (rv32.BclrIOp, 3, 0b1111, 0b0111),
        # Clearing a bit that is already clear leaves the value unchanged
        (rv32.BclrIOp, 2, 0b1011, 0b1011),
        (rv32.BextIOp, 3, 0b1000, 1),
        (rv32.BextIOp, 2, 0b1000, 0),
        (rv32.BsetIOp, 3, 0b0001, 0b1001),
        # Setting a bit that is already set leaves the value unchanged
        (rv32.BsetIOp, 3, 0b1001, 0b1001),
        (rv32.BinvIOp, 3, 0b0101, 0b1101),
        (rv32.BinvIOp, 3, 0b1101, 0b0101),
        # The low nibble rotates into the top nibble
        (rv32.RorIOp, 4, 0xB3, 0x3000000B),
        # Rotating by zero leaves the value unchanged
        (rv32.RorIOp, 0, 0xB3, 0xB3),
        (rv32.RorIOp, 1, 1, 0x80000000),
    ],
)
def test_bit_manipulation_py_operation(
    op_type: type[rv32.RV32RdRsImmShiftOperation],
    shamt: int,
    rs1: int,
    expected: int,
):
    a1 = create_ssa_value(riscv.Registers.A1)

    op = op_type(a1, shamt, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(rs1, i32)) == IntegerAttr(expected, i32)


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


def test_effect_traits():
    """
    Check effects of operations in the rv32 dialect.
    """
    operations = tuple(rv32.RV32.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 10
    assert not unknown_effects_ops

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert all_effects_trait_types == {
        Pure,
        RegisterAllocatedMemoryEffect,
    }

    register_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }
    no_effects_ops = {op for op in effects_ops if op.has_trait(NoMemoryEffect)}

    assert register_effects_ops == {
        rv32.SlliOp,
        rv32.SrliOp,
        rv32.SraiOp,
        rv32.BclrIOp,
        rv32.BextIOp,
        rv32.BinvIOp,
        rv32.BsetIOp,
        rv32.RorIOp,
        rv32.LiOp,
    }
    assert no_effects_ops == {
        rv32.GetRegisterOp,
    }
