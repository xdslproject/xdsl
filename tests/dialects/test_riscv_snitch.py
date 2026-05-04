import pytest

from xdsl import ir, irdl
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, riscv_snitch, snitch
from xdsl.dialects.riscv import RISCVInstruction
from xdsl.traits import (
    EffectInstance,
    HasInsnRepresentation,
    MemoryEffect,
    MemoryEffectKind,
    MemoryReadEffect,
    MemoryWriteEffect,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    get_effects,
)
from xdsl.utils.test_value import create_ssa_value

ground_truth = {
    "dmsrc": ".insn r 0x2b, 0, 0, x0, {0}, {1}",
    "dmdst": ".insn r 0x2b, 0, 1, x0, {0}, {1}",
    "dmcpyi": ".insn r 0x2b, 0, 2, {0}, {1}, {2}",
    "dmcpy": ".insn r 0x2b, 0, 3, {0}, {1}, {2}",
    "dmstati": ".insn r 0x2b, 0, 4, {0}, {1}, {2}",
    "dmstat": ".insn r 0x2b, 0, 5, {0}, {1}, {2}",
    "dmstr": ".insn r 0x2b, 0, 6, x0, {0}, {1}",
    "dmrep": ".insn r 0x2b, 0, 7, x0, {0}, x0",
}


@pytest.mark.parametrize(
    "op",
    [
        riscv_snitch.DMSourceOp,
        riscv_snitch.DMDestinationOp,
        riscv_snitch.DMCopyImmOp,
        riscv_snitch.DMCopyOp,
        riscv_snitch.DMStatImmOp,
        riscv_snitch.DMStatOp,
        riscv_snitch.DMStrideOp,
        riscv_snitch.DMRepOp,
    ],
)
def test_insn_repr(op: RISCVInstruction):
    trait = op.get_trait(HasInsnRepresentation)
    assert trait is not None
    assert trait.get_insn(op) == ground_truth[op.name[13:]]


def test_frep_recursive_effects():

    @irdl.irdl_op_definition
    class TestInstr(irdl.IRDLOperation):
        name = "test.instr"

        rd = irdl.result_def(riscv.FloatRegisterType)
        rs1 = irdl.operand_def(riscv.FloatRegisterType)
        rs2 = irdl.operand_def(riscv.FloatRegisterType)

        traits = irdl.traits_def(RegisterAllocatedMemoryEffect())

    block = ir.Block(
        arg_types=(riscv.Registers.UNALLOCATED_FLOAT, riscv.Registers.UNALLOCATED_FLOAT)
    )

    with ImplicitBuilder(block) as (a, b):
        inner_op = TestInstr.build(
            operands=(a, b), result_types=(riscv.Registers.UNALLOCATED_FLOAT,)
        )
        riscv_snitch.FrepYieldOp(inner_op.rd, a)

    iters = create_ssa_value(riscv.Registers.UNALLOCATED_INT)
    iter_args = (
        create_ssa_value(riscv.Registers.UNALLOCATED_FLOAT),
        create_ssa_value(riscv.Registers.UNALLOCATED_FLOAT),
    )

    frep_op = riscv_snitch.FrepOuterOp(iters, (block,), iter_args)

    # Verify for completeness
    frep_op.verify()

    assert get_effects(inner_op) == set()
    assert get_effects(frep_op) == set()

    TestInstr.traits.add_trait(MemoryReadEffect())

    assert get_effects(inner_op) == {EffectInstance(MemoryEffectKind.READ)}
    assert get_effects(frep_op) == {EffectInstance(MemoryEffectKind.READ)}


def test_exclude_registers():
    read_op = riscv_snitch.ReadOp(
        create_ssa_value(snitch.ReadableStreamType(riscv.Registers.A0))
    )
    read_op.verify()
    assert set(read_op.iter_excluded_registers()) == {
        riscv.Registers.FT0,
        riscv.Registers.FT1,
        riscv.Registers.FT2,
    }

    write_op = riscv_snitch.WriteOp(
        create_ssa_value(riscv.Registers.A0),
        create_ssa_value(snitch.WritableStreamType(riscv.Registers.A0)),
    )
    write_op.verify()
    assert set(write_op.iter_excluded_registers()) == {
        riscv.Registers.FT0,
        riscv.Registers.FT1,
        riscv.Registers.FT2,
    }


def test_effect_traits():
    """
    Check effects of operations in the riscv_snitch dialect.
    """
    operations = tuple(riscv_snitch.RISCV_Snitch.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 26
    assert not unknown_effects_ops

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert all_effects_trait_types == {
        MemoryReadEffect,
        MemoryWriteEffect,
        NoMemoryEffect,
        RecursiveMemoryEffect,
        RegisterAllocatedMemoryEffect,
    }

    memory_read_effects_ops = {
        op for op in effects_ops if op.has_trait(MemoryReadEffect)
    }
    memory_write_effects_ops = {
        op for op in effects_ops if op.has_trait(MemoryWriteEffect)
    }
    recursive_memory_effects_ops = {
        op for op in effects_ops if op.has_trait(RecursiveMemoryEffect)
    }
    register_allocated_memory_effects_ops = {
        op for op in effects_ops if op.has_trait(RegisterAllocatedMemoryEffect)
    }
    no_effects_ops = {op for op in effects_ops if op.has_trait(NoMemoryEffect)}

    assert memory_read_effects_ops == {
        riscv_snitch.DMCopyImmOp,
        riscv_snitch.DMCopyOp,
        riscv_snitch.DMStatImmOp,
        riscv_snitch.DMStatOp,
        riscv_snitch.ReadOp,
    }
    assert memory_write_effects_ops == {
        riscv_snitch.DMCopyImmOp,
        riscv_snitch.DMCopyOp,
        riscv_snitch.DMDestinationOp,
        riscv_snitch.DMDestinationOp,
        riscv_snitch.DMRepOp,
        riscv_snitch.DMSourceOp,
        riscv_snitch.DMStrideOp,
        riscv_snitch.ReadOp,
        riscv_snitch.ScfgwiOp,
        riscv_snitch.ScfgwOp,
        riscv_snitch.WriteOp,
    }
    assert recursive_memory_effects_ops == {
        riscv_snitch.FrepOuterOp,
        riscv_snitch.FrepInnerOp,
    }
    assert register_allocated_memory_effects_ops == {
        riscv_snitch.DMCopyImmOp,
        riscv_snitch.DMCopyOp,
        riscv_snitch.DMStatImmOp,
        riscv_snitch.DMStatOp,
        riscv_snitch.VFAddHOp,
        riscv_snitch.VFAddSOp,
        riscv_snitch.VFCpkASSOp,
        riscv_snitch.VFMacSOp,
        riscv_snitch.VFMacSOp,
        riscv_snitch.VFMaxSOp,
        riscv_snitch.VFMulHOp,
        riscv_snitch.VFMulSOp,
        riscv_snitch.VFSubHOp,
        riscv_snitch.VFSubSOp,
        riscv_snitch.VFSumSOp,
    }
    assert no_effects_ops == {
        riscv_snitch.FrepYieldOp,
        riscv_snitch.GetStreamOp,
    }
