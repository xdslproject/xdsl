import pytest

from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import riscv, rv64
from xdsl.dialects.builtin import (
    IntegerAttr,
    Signedness,
    i64,
)
from xdsl.dialects.riscv.attrs import si12
from xdsl.traits import (
    ConstantLike,
    MemoryEffect,
    MemoryReadEffect,
    MemoryWriteEffect,
    NoMemoryEffect,
    Pure,
)
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_immediate_pseudo_inst():
    lb, ub = Signedness.SIGNLESS.value_range(64)
    assert ub == 18446744073709551616
    assert lb == -9223372036854775808

    # Pseudo-Instruction with custom handling
    with pytest.raises(VerifyException):
        rv64.LiOp(ub, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv64.LiOp(lb - 1, rd=riscv.Registers.A0)

    rv64.LiOp(ub - 1, rd=riscv.Registers.A0)
    rv64.LiOp(lb, rd=riscv.Registers.A0)


def test_immediate_shift_inst():
    # Shift instructions (SLLI, SRLI, SRAI) - 6-bits immediate for RV64
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv64.SlliOp(a1, 1 << 6, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        rv64.SlliOp(a1, -1, rd=riscv.Registers.A0)

    rv64.SlliOp(a1, (1 << 6) - 1, rd=riscv.Registers.A0)


@pytest.mark.parametrize(
    "op_type",
    [rv64.BclrIOp, rv64.BextIOp, rv64.BsetIOp, rv64.BinvIOp, rv64.RorIOp],
)
def test_immediate_bit_manipulation_inst(
    op_type: type[rv64.RV64RdRsImmShiftOperation],
):
    # Bit manipulation instructions - 6-bits immediate for RV64
    a1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        op_type(a1, 1 << 6, rd=riscv.Registers.A0)

    with pytest.raises(VerifyException):
        op_type(a1, -1, rd=riscv.Registers.A0)

    op_type(a1, (1 << 6) - 1, rd=riscv.Registers.A0)


def test_bclri_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv64.BclrIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1111, i64)) == IntegerAttr(0b0111, i64)

    # Clearing a bit that is already clear leaves the value unchanged
    op = rv64.BclrIOp(a1, 2, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1011, i64)) == IntegerAttr(0b1011, i64)


def test_bexti_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv64.BextIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1000, i64)) == IntegerAttr(1, i64)

    op = rv64.BextIOp(a1, 2, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b1000, i64)) == IntegerAttr(0, i64)


def test_bseti_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv64.BsetIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b0001, i64)) == IntegerAttr(0b1001, i64)

    # Setting a bit that is already set leaves the value unchanged
    assert op.py_operation(IntegerAttr(0b1001, i64)) == IntegerAttr(0b1001, i64)


def test_binvi_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    op = rv64.BinvIOp(a1, 3, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0b0101, i64)) == IntegerAttr(0b1101, i64)
    assert op.py_operation(IntegerAttr(0b1101, i64)) == IntegerAttr(0b0101, i64)


def test_rori_py_operation():
    a1 = create_ssa_value(riscv.Registers.A1)

    # The low nibble rotates into the top nibble
    op = rv64.RorIOp(a1, 4, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0xB3, i64)) == IntegerAttr(
        0x300000000000000B, i64
    )

    # Rotating by zero leaves the value unchanged
    op = rv64.RorIOp(a1, 0, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(0xB3, i64)) == IntegerAttr(0xB3, i64)

    op = rv64.RorIOp(a1, 1, rd=riscv.Registers.A0)
    assert op.py_operation(IntegerAttr(1, i64)) == IntegerAttr(0x8000000000000000, i64)


def test_get_constant_value():
    # Test 64-bit LiOp
    li_op = rv64.LiOp(1)
    li_val = get_constant_value(li_op.rd)
    assert li_val == IntegerAttr(1, 64)
    # LiOp implements ConstantLikeInterface so it also has a get_constant_value method:
    constantlike = li_op.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(li_op.rd) == IntegerAttr(1, 64)

    # Test 64-bit zero register
    zero_op = rv64.GetRegisterOp(riscv.Registers.ZERO)
    zero_val = get_constant_value(zero_op.res)
    assert zero_val == IntegerAttr(0, 64)


def test_ld_op_construction():
    lb, ub = Signedness.SIGNED.value_range(12)
    rs1 = create_ssa_value(riscv.Registers.A1)

    with pytest.raises(VerifyException):
        rv64.LdOp(rs1, ub)

    with pytest.raises(VerifyException):
        rv64.LdOp(rs1, lb - 1)

    rv64.LdOp(rs1, ub - 1)
    rv64.LdOp(rs1, lb)
    rv64.LdOp(rs1, 0)

    ld = rv64.LdOp(rs1, 8, rd=riscv.Registers.A0)
    assert ld.rd.type == riscv.Registers.A0
    assert ld.immediate == IntegerAttr(8, si12)


def test_sd_op_construction():
    lb, ub = Signedness.SIGNED.value_range(12)
    rs1 = create_ssa_value(riscv.Registers.A1)
    rs2 = create_ssa_value(riscv.Registers.A2)

    with pytest.raises(VerifyException):
        rv64.SdOp(rs1, rs2, ub)

    with pytest.raises(VerifyException):
        rv64.SdOp(rs1, rs2, lb - 1)

    rv64.SdOp(rs1, rs2, ub - 1)
    rv64.SdOp(rs1, rs2, lb)
    rv64.SdOp(rs1, rs2, 0)

    sd = rv64.SdOp(rs1, rs2, 8)
    assert sd.immediate == IntegerAttr(8, si12)


def test_effect_traits():
    """
    Check effects of operations in the rv64 dialect.
    """
    operations = tuple(rv64.RV64.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 14
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
        MemoryReadEffect,
        MemoryWriteEffect,
    }

    register_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }
    read_effects_ops = {op for op in effects_ops if op.has_trait(MemoryReadEffect)}
    write_effects_ops = {op for op in effects_ops if op.has_trait(MemoryWriteEffect)}
    no_effects_ops = {op for op in effects_ops if op.has_trait(NoMemoryEffect)}

    # RISCVInstruction base adds RegisterAllocatedMemoryEffect to all instructions
    assert register_effects_ops == {
        rv64.SlliOp,
        rv64.SrliOp,
        rv64.SraiOp,
        rv64.SlliwOp,
        rv64.SrliwOp,
        rv64.BclrIOp,
        rv64.BextIOp,
        rv64.BinvIOp,
        rv64.BsetIOp,
        rv64.RorIOp,
        rv64.LiOp,
        rv64.LdOp,
        rv64.SdOp,
    }
    assert read_effects_ops == {rv64.LdOp}
    assert write_effects_ops == {rv64.SdOp}
    assert no_effects_ops == {rv64.GetRegisterOp}
