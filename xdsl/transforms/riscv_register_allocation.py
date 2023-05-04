from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass


class _RegisterIndex:
    idx: int

    def __init__(self, idx: int):
        self.idx = idx


def _allocate_registers(op: Operation, idx: _RegisterIndex) -> None:
    if not isinstance(op, RISCVOp):
        # Don't perform register allocations on non-RISCV-ops
        return

    for result in op.results:
        assert isinstance(result.typ, RegisterType)
        if result.typ.data.name is None:
            result.typ = RegisterType(Register(f"j{idx.idx}"))
            idx.idx += 1


def allocate_registers(op: ModuleOp) -> None:
    """
    Allocates unallocated registers in the module. Currently sets them to an infinite set
    of `j` registers.
    """

    idx = _RegisterIndex(0)

    op.walk(lambda op: _allocate_registers(op, idx))


class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module. Currently sets them to an infinite set
    of `j` registers.
    """

    name = "riscv-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        allocate_registers(op)
