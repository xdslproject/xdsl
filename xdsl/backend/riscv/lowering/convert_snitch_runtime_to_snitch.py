from xdsl.dialects import builtin, riscv, snitch, snitch_runtime
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

# This file is implementing the definitions from the official snitch runtime repo:
# https://github.com/pulp-platform/snitch_cluster/blob/8f8bf58ffa0e2cbee96d30a958da8c6bf53c07a8/sw/snRuntime/src/ssr.h


class LowerSnrtSsrRead(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrReadOp, rewriter: PatternRewriter, /
    ):
        # implementation in C:
        #
        # /// Start a streaming read.
        # inline void snrt_ssr_read(enum snrt_ssr_dm dm, enum snrt_ssr_dim dim,
        #                           volatile void *ptr) {
        #     write_ssr_cfg(REG_RPTR + dim, dm, (uintptr_t)ptr);
        # }
        dim = op.dim
        assert isinstance(dim.owner, riscv.LiOp)
        dim_v = dim.owner.immediate
        assert isinstance(dim_v, builtin.IntegerAttr)

        rewriter.replace_matched_op(
            [snitch.SsrSetDimensionSourceOp(op.dm, op.ptr, dim_v)]
        )


class LowerSnrtSsrWrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrWriteOp, rewriter: PatternRewriter, /
    ):
        # implementation in C:
        #
        # /// Start a streaming write.
        # inline void snrt_ssr_write(enum snrt_ssr_dm dm, enum snrt_ssr_dim dim,
        #                            volatile void *ptr) {
        #     write_ssr_cfg(REG_WPTR + dim, dm, (uintptr_t)ptr);
        # }
        dim = op.dim
        assert isinstance(dim.owner, riscv.LiOp)
        dim_v = dim.owner.immediate
        assert isinstance(dim_v, builtin.IntegerAttr)

        rewriter.replace_matched_op(
            [snitch.SsrSetDimensionDestinationOp(op.dm, op.ptr, dim_v)]
        )


class LowerSnrtSsrEnable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrEnableOp, rewriter: PatternRewriter, /
    ):
        # implementation in C:
        #
        # /// Start a streaming write.
        # inline void snrt_ssr_write(enum snrt_ssr_dm dm, enum snrt_ssr_dim dim,
        #                            volatile void *ptr) {
        #     write_ssr_cfg(REG_WPTR + dim, dm, (uintptr_t)ptr);
        # }
        rewriter.replace_matched_op(snitch.SsrEnable())


class LowerSnrtSsrDisable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrDisableOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(snitch.SsrDisable())


class LowerSnrtLoop2d(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrLoop2dOp, rewriter: PatternRewriter, /
    ):
        # reference implementation:
        # // Configure an SSR data mover for a 2D loop nest.
        #
        # inline void snrt_ssr_loop_2d(enum snrt_ssr_dm dm, size_t b0, size_t b1,
        #                              size_t s0, size_t s1) {
        #     --b0;
        #     --b1;
        #     write_ssr_cfg(REG_BOUNDS + 0, dm, b0);
        #     write_ssr_cfg(REG_BOUNDS + 1, dm, b1);
        #     size_t a = 0;         // <- useless?
        #     write_ssr_cfg(REG_STRIDES + 0, dm, s0 - a);
        #     a += s0 * b0;
        #     write_ssr_cfg(REG_STRIDES + 1, dm, s1 - a);
        #     a += s1 * b1;         // <- equally useless?
        # }
        assert len(op.bounds) == 2
        assert len(op.strides) == 2
        dm = op.data_mover
        int_0 = builtin.IntegerAttr(0, 32)
        int_1 = builtin.IntegerAttr(1, 32)

        rewriter.replace_matched_op(
            [
                new_b0 := riscv.AddiOp(op.bounds[0], -1),
                new_b1 := riscv.AddiOp(op.bounds[1], -1),
                snitch.SsrSetDimensionBoundOp(dm, new_b0, int_0),
                snitch.SsrSetDimensionBoundOp(dm, new_b1, int_1),
                snitch.SsrSetDimensionStrideOp(dm, op.strides[0], int_0),
                s0_b0 := riscv.MulOp(op.bounds[0], op.strides[0]),
                stride_1 := riscv.SubOp(op.strides[1], s0_b0),
                snitch.SsrSetDimensionStrideOp(dm, stride_1, int_1),
            ]
        )


class SnitchRuntimeCallsToSnicthPass(ModulePass):
    name = "convert_snitch_runtime_to_snitch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerSnrtSsrEnable(),
                    LowerSnrtSsrDisable(),
                    LowerSnrtSsrRead(),
                    LowerSnrtSsrWrite(),
                    LowerSnrtLoop2d(),
                ]
            )
        ).rewrite_module(op)
