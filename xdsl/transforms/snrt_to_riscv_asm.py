from xdsl.dialects import snitch_runtime, riscv, builtin
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import MLContext, Operation
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, op_type_rewrite_pattern, GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.passes import ModulePass


class LowerClusterHWBarrier(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch_runtime.ClusterHwBarrierOp, rewriter: PatternRewriter, /):
        """
        Lowers to:

        /// Synchronize cores in a cluster with a hardware barrier
        inline void snrt_cluster_hw_barrier() {
            asm volatile("csrr x0, 0x7C2" ::: "memory");
        }
        """
        rewriter.replace_matched_op([
            zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
            riscv.CsrrsOp(
                rd=riscv.Registers.ZERO,
                rs1=zero,
                csr=IntegerAttr(0x7C2, 12),
            )
        ], [])


class ConvertSnrtToRISCV(ModulePass):
    name = "convert-snrt-to-riscv-asm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([
            LowerClusterHWBarrier(),
        ])).rewrite_module(op)