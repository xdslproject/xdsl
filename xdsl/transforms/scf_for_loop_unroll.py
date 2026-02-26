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


class UnrollLoopPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        if (
            not isinstance(lb_op := op.lb.owner, arith.ConstantOp)
            or not isinstance(ub_op := op.ub.owner, arith.ConstantOp)
            or not isinstance(step_op := op.step.owner, arith.ConstantOp)
        ):
            return

        assert isinstance(lb_op.value, builtin.IntegerAttr)
        assert isinstance(ub_op.value, builtin.IntegerAttr)
        assert isinstance(step_op.value, builtin.IntegerAttr)

        lb = lb_op.value.value.data
        ub = ub_op.value.value.data
        step = step_op.value.value.data

        iter_args: tuple[SSAValue, ...] = op.iter_args

        i_arg, *block_iter_args = op.body.block.args

        for i in range(lb, ub, step):
            i_op = rewriter.insert(
                arith.ConstantOp(builtin.IntegerAttr(i, lb_op.value.type))
            )
            i_op.result.name_hint = i_arg.name_hint

            value_mapper: dict[SSAValue, SSAValue] = {
                arg: val for arg, val in zip(block_iter_args, iter_args, strict=True)
            }
            value_mapper[i_arg] = i_op.result

            for inner_op in op.body.block.ops:
                if isinstance(inner_op, scf.YieldOp):
                    iter_args = tuple(
                        value_mapper.get(val, val) for val in inner_op.arguments
                    )
                else:
                    rewriter.insert(inner_op.clone(value_mapper))

        rewriter.replace_op(op, (), iter_args)


class ScfForLoopUnrollPass(ModulePass):
    """
    Fully unrolls all loops where the lb, ub, and step are constants.
    """

    name = "scf-for-loop-unroll"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(UnrollLoopPattern()).rewrite_module(op)
