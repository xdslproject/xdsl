from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, scf
from xdsl.ir import BlockArgument, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def is_foldable(val: SSAValue, for_op: scf.For):
    if isinstance(val, BlockArgument):
        return True

    if not isinstance(val, OpResult):
        return False

    return not for_op.is_ancestor(val.op)


class ScfForLoopRangeFolding(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter) -> None:
        index = op.body.block.args[0]

        # Fold until a fixed point is reached
        while True:
            if len(index.uses) != 1:
                # If the induction variable is used more than once, we can't fold its
                # arith ops into the loop range
                return

            user = next(iter(index.uses)).operation

            if not isinstance(user, arith.Addi | arith.Muli):
                return

            if user.operands[0] is index:
                if not is_foldable(user.operands[1], op):
                    return
                folding_const = user.operands[1]
            else:
                if not is_foldable(user.operands[0], op):
                    return
                folding_const = user.operands[0]

            new_name_hint = (
                user.result.name_hint if user.result.name_hint is not None else "new"
            )
            match user:
                case arith.Addi():
                    rewriter.insert_op_before_matched_op(
                        [
                            new_lb := arith.Addi(op.lb, folding_const),
                            new_ub := arith.Addi(op.ub, folding_const),
                        ]
                    )
                case arith.Muli():
                    rewriter.insert_op_before_matched_op(
                        [
                            new_lb := arith.Muli(op.lb, folding_const),
                            new_ub := arith.Muli(op.ub, folding_const),
                            new_step := arith.Muli(op.step, folding_const),
                        ]
                    )
                    op.operands[2] = new_step.result
                    new_step.result.name_hint = new_name_hint + "_step"

            new_lb.result.name_hint = new_name_hint + "_lb"
            new_ub.result.name_hint = new_name_hint + "_ub"
            op.operands[0] = new_lb.result
            op.operands[1] = new_ub.result

            rewriter.replace_op(user, [], [index])


class ScfForLoopRangeFoldingPass(ModulePass):
    """
    xdsl implementation of the pass with the same name
    """

    name = "scf-for-loop-range-folding"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ScfForLoopRangeFolding(),
            apply_recursively=True,
        ).rewrite_module(op)
