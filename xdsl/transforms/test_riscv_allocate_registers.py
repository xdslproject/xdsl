from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass, PassPipeline
from xdsl.transforms import riscv_allocate_infinite_registers, riscv_allocate_registers


@dataclass(frozen=True)
class TestRiscvAllocateRegistersPass(ModulePass):
    name = "test-riscv-allocate-registers"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pipeline = PassPipeline(
            (
                riscv_allocate_registers.RISCVAllocateRegistersPass(
                    force_infinite=True
                ),
                riscv_allocate_infinite_registers.RISCVAllocateInfiniteRegistersPass(),
            )
        )
        pipeline.apply(ctx, op)
