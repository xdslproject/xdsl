from xdsl.dialects import riscv, riscv_func
from xdsl.ir import Region
from xdsl.traits import CallableOpInterface, MemoryEffect


def test_callable_interface():
    a0, a1 = riscv.Registers.A0, riscv.Registers.A1
    fa0, fa1 = riscv.Registers.FA0, riscv.Registers.FA1

    region = Region()
    func = riscv_func.FuncOp("callable", region, ((a0, a1), (fa0, fa1)))

    trait = func.get_trait(CallableOpInterface)

    assert trait is not None

    assert trait.get_callable_region(func) is region
    assert trait.get_argument_types(func) == (a0, a1)
    assert trait.get_result_types(func) == (fa0, fa1)


def test_effect_traits():
    """
    Check effects of operations in the riscv_func dialect.
    """
    operations = tuple(riscv_func.RISCV_Func.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 0
    assert len(unknown_effects_ops) == 4

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert not all_effects_trait_types
