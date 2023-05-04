from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


def allocate_registers(op: ModuleOp) -> None:
    """
    Allocates unallocated registers in the module. Currently sets them to an infinite set
    of `j` registers.
    """
    idx: int = 0
    for operation in op.ops:
        # Don't perform register allocations on non-RISCV-ops
        if not isinstance(operation, RISCVOp):
            continue

        for result in operation.results:
            assert isinstance(result.typ, RegisterType)
            if result.typ.data.name is None:
                result.typ = RegisterType(Register(f"j{idx}"))
                idx += 1


class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module. Currently sets them to an infinite set
    of `j` registers.
    """

    name = "riscv-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        allocate_registers(op)
