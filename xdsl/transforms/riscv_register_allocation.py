from dataclasses import dataclass

from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.dialects import riscv, riscv_func
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LoadZeroImmediatePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.LiOp, rewriter: PatternRewriter) -> None:
        if not isinstance(op.immediate, IntegerAttr) or op.immediate.value.data != 0:
            return

        assert isinstance(op.rd.type, riscv.IntRegisterType)

        if op.rd.type.is_allocated:
            return

        # Set the result type to 0
        op.rd.type = riscv.Registers.ZERO

        for use in op.rd.uses:
            if isinstance(use.operation, riscv_scf.ForRofOperation):



@dataclass
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

        # 0 should always be zero
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LoadZeroImmediatePattern()]),
            apply_recursively=False,
        ).rewrite_module(op)

        for inner_op in op.walk():
            if isinstance(inner_op, riscv_func.FuncOp):
                allocator = allocator_strategies[self.allocation_strategy]()
                if self.limit_registers is not None:
                    allocator.available_registers.limit_registers(self.limit_registers)
                allocator.exclude_preallocated = self.exclude_preallocated
                allocator.exclude_snitch_reserved = self.exclude_snitch_reserved
                allocator.allocate_func(inner_op)
