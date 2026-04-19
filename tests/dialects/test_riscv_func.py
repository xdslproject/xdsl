from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.dialects import riscv, riscv_func
from xdsl.ir import Region
from xdsl.traits import (
    CallableOpInterface,
    MemoryAllocEffect,
    MemoryEffect,
    MemoryFreeEffect,
    MemoryReadEffect,
    MemoryWriteEffect,
)


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
    assert effects_ops == {
        riscv_func.CallOp,
        riscv_func.ReturnOp,
        riscv_func.SyscallOp,
    }
    assert unknown_effects_ops == {
        riscv_func.FuncOp,
    }

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert all_effects_trait_types == {
        MemoryAllocEffect,
        MemoryFreeEffect,
        MemoryReadEffect,
        MemoryWriteEffect,
        RegisterAllocatedMemoryEffect,
        riscv_func.RiscvFunctionCallMemoryEffect,
    }

    alloc_effects_ops = {op for op in effects_ops if op.has_trait(MemoryAllocEffect)}
    free_effects_ops = {op for op in effects_ops if op.has_trait(MemoryFreeEffect)}
    read_effects_ops = {op for op in effects_ops if op.has_trait(MemoryReadEffect)}
    write_effects_ops = {op for op in effects_ops if op.has_trait(MemoryWriteEffect)}
    riscv_func_call_effects_ops = {
        op
        for op in effects_ops
        if op.has_trait(riscv_func.RiscvFunctionCallMemoryEffect)
    }

    register_allocated_memory_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }

    assert alloc_effects_ops == {riscv_func.CallOp, riscv_func.SyscallOp}
    assert free_effects_ops == {riscv_func.CallOp, riscv_func.SyscallOp}
    assert read_effects_ops == {riscv_func.CallOp, riscv_func.SyscallOp}
    assert write_effects_ops == {riscv_func.CallOp, riscv_func.SyscallOp}
    assert riscv_func_call_effects_ops == {riscv_func.CallOp, riscv_func.SyscallOp}

    assert register_allocated_memory_effects_ops == {
        riscv_func.CallOp,
        riscv_func.ReturnOp,
    }
