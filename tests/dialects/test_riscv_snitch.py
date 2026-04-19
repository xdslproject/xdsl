import pytest

from xdsl import ir, irdl
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, riscv_snitch
from xdsl.dialects.riscv import RISCVInstruction
from xdsl.traits import (
    EffectInstance,
    HasInsnRepresentation,
    MemoryEffectKind,
    MemoryReadEffect,
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
