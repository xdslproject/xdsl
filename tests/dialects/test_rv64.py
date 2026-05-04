import pytest

from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import riscv, rv64
from xdsl.dialects.builtin import (
    IntegerAttr,
    Signedness,
)
from xdsl.traits import ConstantLike, MemoryEffect, NoMemoryEffect, Pure
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import VerifyException


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


def test_effect_traits():
    """
    Check effects of operations in the rv64 dialect.
    """
    operations = tuple(rv64.RV64.operations)
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
    }

    register_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }
    no_effects_ops = {op for op in effects_ops if op.has_trait(NoMemoryEffect)}

    assert not register_effects_ops
    assert no_effects_ops == {
        rv64.LiOp,  # This is a bug, will fix in an upcoming PR
        rv64.GetRegisterOp,
    }
