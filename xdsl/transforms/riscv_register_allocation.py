from dataclasses import dataclass

from xdsl.backend.riscv.register_allocation import (
    RegisterAllocatorBlockNaive,
    RegisterAllocatorJRegs,
    RegisterAllocatorLivenessBlockNaive,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


@dataclass
class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    allocation_strategy: str = "GlobalJRegs"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        allocator_strategies = {
            "GlobalJRegs": RegisterAllocatorJRegs,
            "BlockNaive": RegisterAllocatorBlockNaive,
            "LivenessBlockNaive": RegisterAllocatorLivenessBlockNaive,
        }

        if self.allocation_strategy not in allocator_strategies:
            raise ValueError(
                f"Unknown register allocation strategy {self.allocation_strategy}. "
                f"Available allocation types: {allocator_strategies.keys()}"
            )

        allocator = allocator_strategies[self.allocation_strategy]()
        allocator.allocate_registers(op)
