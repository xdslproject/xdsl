from dataclasses import dataclass

from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.backend.riscv.riscv_register_queue import RiscvRegisterQueue
from xdsl.context import Context
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

    exclude_snitch_reserved: bool = True
    """Excludes floating-point registers that are used by the Snitch ISA extensions."""

    add_regalloc_stats: bool = False
    """
    Inserts a comment with register allocation info in the IR.
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

        if self.limit_registers is not None and self.limit_registers < 0:
            raise ValueError(
                "The limit of available registers cannot be less than 0."
                "When set to 0 it signifies all available registers are used."
            )

        for inner_op in op.walk():
            if isinstance(inner_op, riscv_func.FuncOp):
                riscv_register_queue = RiscvRegisterQueue.default()
                if self.limit_registers is not None:
                    riscv_register_queue.limit_registers(self.limit_registers)
                allocator = allocator_strategies[self.allocation_strategy](
                    riscv_register_queue
                )
                allocator.exclude_snitch_reserved = self.exclude_snitch_reserved
                allocator.allocate_func(
                    inner_op, add_regalloc_stats=self.add_regalloc_stats
                )
