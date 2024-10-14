from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.bufferization import MaterializeInDestination
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.func import Return
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import VerifyException


def make_materialize_op(source: SSAValue, dest: SSAValue) -> MaterializeInDestination:
    return MaterializeInDestination(operands=[source, dest], result_types=[source.type])


@dataclass
class SubstituteDonatedTensors(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Return, rewriter: PatternRewriter, /):
        func_op = op.parent_op()
        if func_op is None:
            raise VerifyException("Return operation should be tied to a FuncOp")

        arg_attrs = getattr(func_op, "arg_attrs")
        args = getattr(func_op, "args")

        if arg_attrs is None:
            return

        donated_inputs = [
            inp
            for inp, attr in zip(args, arg_attrs)
            if isinstance(inp.type, TensorType) and "tf.aliasing_output" in attr.data
        ]

        value_mapper: dict[SSAValue, SSAValue] = {}
        new_ops: list[Operation] = []
        for output in op.arguments:
            if type(getattr(output, "type")) is not TensorType:
                continue

            for i, arg in enumerate(donated_inputs):
                if getattr(arg, "type").is_same_type_with(output.type):
                    new_ops.append(make_materialize_op(output, donated_inputs.pop(i)))
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
