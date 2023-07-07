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

    limit_registers: int = 0

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

        if self.limit_registers < 0:
            raise ValueError(
                "The limit of available registers cannot be less than 0."
                "When set to 0 it signifies all available registers are used."
            )

        allocator = allocator_strategies[self.allocation_strategy](
            limit_registers=self.limit_registers
        )
        allocator.allocate_registers(op)
