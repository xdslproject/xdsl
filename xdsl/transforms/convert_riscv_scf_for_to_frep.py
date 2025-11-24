from itertools import chain

from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_scf, riscv_snitch, snitch
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import Pure

ALLOWED_FREP_OP_LOWERING_TYPES = (
    *riscv_snitch.ALLOWED_FREP_OP_TYPES,
    riscv_scf.YieldOp,
)


class ScfForLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter) -> None:
        body_block = op.body.block
        indvar = body_block.args[0]
        if indvar.uses:
            # 1. Induction variable is used
            return

        if not (
            isinstance(step_op := op.step.owner, riscv.LiOp)
            and isinstance(step_op.immediate, builtin.IntegerAttr)
            and step_op.immediate.value.data == 1
        ):
            # 2. Step is 1
            return

        if not all(
            isinstance(
                value.type,
                riscv.FloatRegisterType
                | snitch.ReadableStreamType
                | snitch.WritableStreamType,
            )
            for o in body_block.ops
            for value in chain(o.operands, o.results)
        ):
            # 3. All operations in the loop operate on float registers
            return

        if not all(
            isinstance(o, ALLOWED_FREP_OP_LOWERING_TYPES) or o.has_trait(Pure)
            for o in body_block.ops
        ):
            # 4. All operations are pure or one of
            #     a) riscv_snitch.read
            #     b) riscv_snitch.write
            #     c) builtin.unrealized_conversion_cast
            return

        rewriter.erase_block_argument(indvar)
        rewriter.replace_op(
            op,
            (
                iter_count := riscv.SubOp(op.ub, op.lb),
                iter_count_minus_one := riscv.AddiOp(iter_count, -1),
                riscv_snitch.FrepOuterOp(
                    iter_count_minus_one,
                    rewriter.move_region_contents_to_new_regions(op.body),
                    op.iter_args,
                ),
            ),
        )


class ScfYieldLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_scf.YieldOp, rewriter: PatternRewriter
    ) -> None:
        if isinstance(op.parent_op(), riscv_snitch.FRepOperation):
            rewriter.replace_op(op, riscv_snitch.FrepYieldOp(*op.operands))


class ConvertRiscvScfForToFrepPass(ModulePass):
    """
    Converts all riscv_scf.for loops to riscv_snitch.frep_outer loops, if the loops pass
    the riscv_snitch.frep_outer verification criteria:

    1. The induction variable is not used
    2. Step is 1
    3. All operations in the loop all operate on float registers
    4. All operations are pure or one of
        a) riscv_snitch.read
        b) riscv_snitch.write
        c) builtin.unrealized_conversion_cast

    """

    name = "convert-riscv-scf-for-to-frep"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ScfYieldLowering(),
                    ScfForLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
