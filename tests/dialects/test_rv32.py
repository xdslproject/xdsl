import pytest

from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    IntegerAttr,
    Signedness,
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
    assert len(effects_ops) == 2
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
        rv32.LiOp,
    }
    assert no_effects_ops == {
        rv32.GetRegisterOp,
    }
