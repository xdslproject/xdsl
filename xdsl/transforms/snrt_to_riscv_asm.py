from xdsl.dialects import builtin, riscv, snitch_runtime
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerClusterHWBarrier(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterHwBarrierOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to:

        /// Synchronize cores in a cluster with a hardware barrier
        inline void snrt_cluster_hw_barrier() {
            asm volatile("csrr x0, 0x7C2" ::: "memory");
        }
        """
        rewriter.replace_matched_op(
            [
                zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                riscv.CsrrsOp(
                    rd=riscv.Registers.ZERO,
                    rs1=zero,
                    csr=IntegerAttr(0x7C2, 12),
                ),
            ],
            [],
        )


class LowerSSRDisable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrDisableOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to:

            /// Disable SSR.
            inline void snrt_ssr_disable() {
            #ifdef __TOOLCHAIN_LLVM__
                __builtin_ssr_disable();
            #else
                asm volatile("csrci 0x7C0, 1\n");
            #endif
            }

        P.S. This specific rewrite ignores the LLVM case and goes
                straight to the generic one.
        """
        rewriter.replace_matched_op(
            [riscv.CsrrciOp(csr=IntegerAttr(0x7C0, 12), immediate=IntegerAttr(1, 4))],
            [],
        )


class ConvertSnrtToRISCV(ModulePass):
    name = "convert-snrt-to-riscv-asm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerClusterHWBarrier(), LowerSSRDisable()])
        ).rewrite_module(op)
