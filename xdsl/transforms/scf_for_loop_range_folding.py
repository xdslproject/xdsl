from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def is_foldable(val: SSAValue, for_op: scf.ForOp):
    return not for_op.is_ancestor(val.owner)


class ScfForLoopRangeFolding(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        index = op.body.block.args[0]

        # Fold until a fixed point is reached
        while True:
            if not index.has_one_use():
                # If the induction variable is used more than once, we can't fold its
                # arith ops into the loop range
                return

            user = next(iter(index.uses)).operation

            if not isinstance(user, arith.AddiOp | arith.MuliOp):
                return

            if user.operands[0] is index:
                if not is_foldable(user.operands[1], op):
                    return
                folding_const = user.operands[1]
            else:
                if not is_foldable(user.operands[0], op):
                    return
                folding_const = user.operands[0]

            match user:
                case arith.AddiOp():
                    rewriter.insert_op(
                        [
                            new_lb := arith.AddiOp(op.lb, folding_const),
                            new_ub := arith.AddiOp(op.ub, folding_const),
                        ]
                    )
                case arith.MuliOp():
                    rewriter.insert_op(
                        [
                            new_lb := arith.MuliOp(op.lb, folding_const),
                            new_ub := arith.MuliOp(op.ub, folding_const),
                            new_step := arith.MuliOp(op.step, folding_const),
                        ]
                    )
                    op.operands[2] = new_step.result

            op.operands[0] = new_lb.result
            op.operands[1] = new_ub.result

            rewriter.replace_op(user, [], [index])


class ScfForLoopRangeFoldingPass(ModulePass):
    """
    xdsl implementation of the pass with the same name
    """

    name = "scf-for-loop-range-folding"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ScfForLoopRangeFolding(), apply_recursively=False, walk_regions_first=True
        ).rewrite_module(op)
