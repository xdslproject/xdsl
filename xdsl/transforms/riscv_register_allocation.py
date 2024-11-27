from dataclasses import dataclass

from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.context import MLContext
from xdsl.dialects import riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    allocation_strategy: str = "LivenessBlockNaive"

    limit_registers: int | None = None

    exclude_preallocated: bool = True
    """
    Enables tracking of already allocated registers and excludes them from the
    available set.
    This does not keep track of any liveness information and the preallocated registers
    are excluded completely from any further allocation decisions.
    """

    exclude_snitch_reserved: bool = True
    """Excludes floating-point registers that are used by the Snitch ISA extensions."""

    add_regalloc_stats: bool = False
    """
    Inserts a comment with register allocation info in the IR.
    """

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        allocator_strategies = {
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
                allocator = allocator_strategies[self.allocation_strategy]()
                if self.limit_registers is not None:
                    allocator.available_registers.limit_registers(self.limit_registers)
                allocator.exclude_preallocated = self.exclude_preallocated
                allocator.exclude_snitch_reserved = self.exclude_snitch_reserved
                allocator.allocate_func(
                    inner_op, add_regalloc_stats=self.add_regalloc_stats
                )
