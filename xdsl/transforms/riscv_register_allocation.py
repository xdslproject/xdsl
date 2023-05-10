from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


class RegisterAllocator:
    idx: int

    def __init__(self) -> None:
        self.idx = 0

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Allocates unallocated registers in the module. Currently sets them to an infinite set
        of `j` registers.
        """
        for op in module.walk():
            if not isinstance(op, RISCVOp):
                # Don't perform register allocations on non-RISCV-ops
                continue

            for result in op.results:
                assert isinstance(result.typ, RegisterType)
                if result.typ.data.name is None:
                    result.typ = RegisterType(Register(f"j{self.idx}"))
                    self.idx += 1


class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module. Currently sets them to an infinite set
    of `j` registers.
    """

    name = "riscv-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        allocator = RegisterAllocator()
        allocator.allocate_registers(op)
