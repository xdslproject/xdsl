# from copy import deepcopy
import math

from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

#  This pass unrolls loops
#
#  Similar to LLVM's loop unrolling:
#  https://mlir.llvm.org/doxygen/namespacemlir.html#a7d85acf663c85ff9286024bd8ca84f1b
#


class UnrollLoopsPattern(RewritePattern):
    unroll_factor: int = 4

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        """Unroll simple scf for loops with integer bounds.

        This will duplicate the inner loop body (and forward referenced variables)
        `unroll_factor` times, then multiply the loop step by `unroll_factor`.
        If the unroll factor does not cleanly divide the loop range, don't match
        to avoid needing a clean-up loop.

        Similar to <https://mlir.llvm.org/doxygen/namespacemlir.html#a7d85acf663c85ff9286024bd8ca84f1b>
        """
        if self.unroll_factor == 1:
            return
        if len(op.body.ops) == 0:
            return

        ## First check if we can unroll this loop
        lb: arith.ConstantOp = op.lb.op  # pyright: ignore
        ub: arith.ConstantOp = op.ub.op  # pyright: ignore
        step: arith.ConstantOp = op.step.op  # pyright: ignore

        # Ensure we have constant bounds and step
        if not (
            lb.has_trait(arith.ConstantLike)
            and ub.has_trait(arith.ConstantLike)
            and step.has_trait(arith.ConstantLike)
        ):
            return
        lb_value: int = lb.value.value.data  # pyright: ignore
        ub_value: int = ub.value.value.data  # pyright: ignore
        step_value: int = step.value.value.data  # pyright: ignore

        # Check if the unroll factor evenly divides the number of iterations
        trip_count = math.ceil((ub_value - lb_value) / step_value)
        if trip_count % self.unroll_factor != 0:
            return

        # Update the step size
        new_step = arith.ConstantOp(
            builtin.IntegerAttr(step_value * self.unroll_factor, step.result.type)
        )
        rewriter.replace_op(step, new_step)

        # Get the loop body block
        body_block = op.body.block
        original_loop_ops = list(body_block.ops)

        prev_yielded = None
        for orig_op in body_block.ops:
            if isinstance(orig_op, scf.YieldOp):
                prev_yielded = orig_op.clone()
                rewriter.erase_op(orig_op)

        # Duplicate the body multiple times based on the unroll factor
        for i in range(1, self.unroll_factor):
            for dup_op in original_loop_ops:
                if isinstance(dup_op, scf.YieldOp):
                    prev_yielded = dup_op.clone()
                    break

                new_op = dup_op.clone()
                print(new_op)
                for old, new in zip(op.iter_args, prev_yielded.arguments):
                    print(old, new)
                    print(old)
                    print(new)
                    # old.replace_by(new)
                # for operand in new_op.operands:
                #     print(operand.owner)
                print()
                rewriter.insert_op(new_op, InsertPoint.after(body_block.last_op))

        rewriter.insert_op(prev_yielded.clone(), InsertPoint.after(body_block.last_op))


class ScfForLoopUnrollPass(ModulePass):
    """
    Unrolls loops
    """

    name = "scf-for-loop-unroll"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(UnrollLoopsPattern()).rewrite_module(op)
