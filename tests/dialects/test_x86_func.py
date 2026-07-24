from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import x86_func
from xdsl.traits import MemoryEffect


def test_effect_traits():
    """
    Check effects of operations in the x86_func dialect.
    """
    operations = tuple(x86_func.X86_FUNC.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 1
    assert len(unknown_effects_ops) == 1

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert all_effects_trait_types == {
        RegisterAllocatedMemoryEffect,
    }

    register_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }

    assert register_effects_ops == {
        x86_func.RetOp,
    }
