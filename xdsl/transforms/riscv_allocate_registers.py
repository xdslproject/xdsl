from dataclasses import dataclass

from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class RISCVAllocateRegistersPass(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    allocation_strategy: str = "LivenessBlockNaive"

    add_regalloc_stats: bool = False
    """
    Inserts a comment with register allocation info in the IR.
    """

    allow_infinite: bool = False
    """
    Whether to allow using infinite registers during register allocation.
    """

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        allocator_strategies = {
            "LivenessBlockNaive": RegisterAllocatorLivenessBlockNaive,
        }

        if self.allocation_strategy not in allocator_strategies:
            raise ValueError(
                f"Unknown register allocation strategy {self.allocation_strategy}. "
                f"Available allocation types: {allocator_strategies.keys()}"
            )

        for inner_op in op.walk():
            if isinstance(inner_op, riscv_func.FuncOp):
                register_stack = RiscvRegisterStack.get(
                    allow_infinite=self.allow_infinite
                )
                allocator = allocator_strategies[self.allocation_strategy](
                    register_stack
                )
                allocator.allocate_func(
                    inner_op, add_regalloc_stats=self.add_regalloc_stats
                )
