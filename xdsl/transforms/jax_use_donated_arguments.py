from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.bufferization import MaterializeInDestinationOp
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class SubstituteDonatedTensors(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        func_op = op.parent_op()
        assert isinstance(func_op, FuncOp)

        if func_op.arg_attrs is None:
            return

        donated_inputs = [
            inp
            for inp, attr in zip(func_op.args, func_op.arg_attrs, strict=True)
            if isinstance(inp.type, TensorType) and "tf.aliasing_output" in attr.data
        ]

        value_mapper: dict[SSAValue, SSAValue] = {}
        new_ops: list[Operation] = []
        for output in op.arguments:
            for i, arg in enumerate(donated_inputs):
                if arg.type == output.type:
                    new_ops.append(
                        MaterializeInDestinationOp(
                            operands=[output, donated_inputs.pop(i)],
                            result_types=[output.type],
                        )
                    )
                    value_mapper[output] = new_ops[-1].results[0]
                    break

        new_ops.append(op.clone(value_mapper))
        rewriter.replace_matched_op(new_ops)


@dataclass(frozen=True)
class JaxUseDonatedArguments(ModulePass):
    name = "jax-use-donated-arguments"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([SubstituteDonatedTensors()]),
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=True,
        )
        the_one_pass.rewrite_module(op)
