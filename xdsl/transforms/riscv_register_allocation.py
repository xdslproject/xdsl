from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


class RISCVRegisterAllocation(ModulePass):
    name = "riscv-allocate-registers"

    def __init__(self) -> None:
        super().__init__()
        self.idx: int = 0

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        for operation in op.ops:
            # Don't perform register allocations on non-RISCV-ops
            if not isinstance(operation, RISCVOp):
                continue
            else:
                for result in operation.results:
                    assert isinstance(result.typ, RegisterType)
                    if result.typ.data.name is None:
                        result.typ = RegisterType(Register(f"j{self.idx}"))
                self.idx += 1
