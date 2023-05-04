from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
)


class AllocateRegisters(RewritePattern):
    def __init__(self) -> None:
        super().__init__()
        self.idx: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: RISCVOp,
        rewriter: PatternRewriter,
    ):
        for result in op.results:
            assert isinstance(result.typ, RegisterType)
            if result.typ.data.name is None:
                result.typ = RegisterType(Register(f"j{self.idx}"))
                self.idx += 1


class RISCVRegisterAllocation(ModulePass):
    name = "riscv-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            AllocateRegisters(), apply_recursively=True, walk_reverse=False
        )
        walker.rewrite_module(op)
