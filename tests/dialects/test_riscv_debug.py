from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import riscv_debug
from xdsl.traits import (
    EffectInstance,
    MemoryEffect,
    MemoryEffectKind,
    MemoryWriteEffect,
    get_effects,
)


def test_effects():
    op = riscv_debug.PrintfOp("hello")

    assert get_effects(op) == {EffectInstance(MemoryEffectKind.WRITE)}


def test_effect_traits():
    """
    Check effects of operations in the risv_debug dialect.
    """
    operations = tuple(riscv_debug.RISCV_Debug.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 1
    assert not unknown_effects_ops

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert all_effects_trait_types == {
        MemoryWriteEffect,
        RegisterAllocatedMemoryEffect,
    }

    register_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }
    write_effects_ops = {op for op in effects_ops if op.has_trait(MemoryWriteEffect)}

    assert register_effects_ops == {riscv_debug.PrintfOp}
    assert write_effects_ops == {riscv_debug.PrintfOp}
