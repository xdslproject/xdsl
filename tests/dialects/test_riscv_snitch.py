import pytest

from xdsl.dialects import riscv_snitch
from xdsl.dialects.riscv import RISCVInstruction
from xdsl.traits import HasInsnRepresentation

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
    # Limitation of Pyright, see https://github.com/microsoft/pyright/issues/7105
    # We are currently stuck on an older version of Pyright, the update is
    # tracked in https://github.com/xdslproject/xdsl/issues/2791
    assert trait.get_insn(op) == ground_truth[op.name[13:]]
