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


def test_immediate_bclri_inst():
    # BCLRI - 5-bits immediate, shamt[5]=1 encodings are reserved on RV32
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.BclrIOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.BclrIOp(a1, -1, rd=riscv.Registers.A0)

    rv32.BclrIOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_immediate_bexti_inst():
    # BEXTI - 5-bits immediate, shamt[5]=1 encodings are reserved on RV32
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.BextIOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.BextIOp(a1, -1, rd=riscv.Registers.A0)

    rv32.BextIOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_immediate_bseti_inst():
    # BSETI - 5-bits immediate, shamt[5]=1 encodings are reserved on RV32
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.BsetIOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.BsetIOp(a1, -1, rd=riscv.Registers.A0)

    rv32.BsetIOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_immediate_binvi_inst():
    # BINVI - 5-bits immediate, shamt[5]=1 encodings are reserved on RV32
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.BinvIOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.BinvIOp(a1, -1, rd=riscv.Registers.A0)

    rv32.BinvIOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_immediate_rori_inst():
    # RORI - 5-bits immediate, shamt[5]=1 encodings are reserved on RV32
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv32.RorIOp(a1, 1 << 5, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv32.RorIOp(a1, -1, rd=riscv.Registers.A0)

    rv32.RorIOp(a1, (1 << 5) - 1, rd=riscv.Registers.A0)


def test_bclri_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv32.BclrIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1111, i32)) == IntegerAttr(0b0111, i32)

    # Clearing a bit that is already clear leaves the value unchanged
    op = rv32.BclrIOp(a1, 2, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1011, i32)) == IntegerAttr(0b1011, i32)


def test_bexti_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv32.BextIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1000, i32)) == IntegerAttr(1, i32)

    op = rv32.BextIOp(a1, 2, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1000, i32)) == IntegerAttr(0, i32)


def test_bseti_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv32.BsetIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b0001, i32)) == IntegerAttr(0b1001, i32)

    # Setting a bit that is already set leaves the value unchanged
    assert op.py_operation(IntegerAttr(0b1001, i32)) == IntegerAttr(0b1001, i32)


def test_binvi_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv32.BinvIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b0101, i32)) == IntegerAttr(0b1101, i32)
    assert op.py_operation(IntegerAttr(0b1101, i32)) == IntegerAttr(0b0101, i32)


def test_rori_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    # The low nibble rotates into the top nibble
    op = rv32.RorIOp(a1, 4, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0xB3, i32)) == IntegerAttr(0x3000000B, i32)

    # Rotating by zero leaves the value unchanged
    op = rv32.RorIOp(a1, 0, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0xB3, i32)) == IntegerAttr(0xB3, i32)

    op = rv32.RorIOp(a1, 1, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(1, i32)) == IntegerAttr(0x80000000, i32)


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
