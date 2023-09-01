from dataclasses import dataclass

from xdsl.backend.riscv.register_allocation import (
    RegisterAllocatorBlockNaive,
    RegisterAllocatorLivenessBlockNaive,
)
from xdsl.dialects import riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


@dataclass
class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    allocation_strategy: str = "BlockNaive"

    limit_registers: int | None = None

    exclude_preallocated: bool = False

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        allocator_strategies = {
            "BlockNaive": RegisterAllocatorBlockNaive,
            "LivenessBlockNaive": RegisterAllocatorLivenessBlockNaive,
        }

        if self.allocation_strategy not in allocator_strategies:
            raise ValueError(
                f"Unknown register allocation strategy {self.allocation_strategy}. "
                f"Available allocation types: {allocator_strategies.keys()}"
            )

        if self.limit_registers is not None and self.limit_registers < 0:
            raise ValueError(
                "The limit of available registers cannot be less than 0."
                "When set to 0 it signifies all available registers are used."
            )

        for inner_op in op.walk():
            if isinstance(inner_op, riscv_func.FuncOp):
                allocator = allocator_strategies[self.allocation_strategy](
                    limit_registers=self.limit_registers,
                    exclude_preallocated=self.exclude_preallocated,
                )
                allocator.allocate_func(inner_op)
