from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.bufferization import MaterializeInDestinationOp
from xdsl.dialects.builtin import Attribute, FunctionType, TensorType
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
    remove_matched_outputs: bool = False

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

        if len(donated_inputs) == 0:
            return

        new_ops: list[Operation] = []
        new_outputs: list[SSAValue] = []
        matched_output_idxes: set[int] = set()

        for output_idx, output in enumerate(op.arguments):
            final_output = output
            for i, arg in enumerate(donated_inputs):
                if arg.type == output.type:
                    new_ops.append(
                        MaterializeInDestinationOp(
                            operands=[output, donated_inputs.pop(i)],
                            result_types=[output.type],
                        )
                    )
                    final_output = new_ops[-1].results[0]
                    matched_output_idxes.add(output_idx)
                    break
            new_outputs.append(final_output)

        output_types = list(func_op.function_type.outputs.data)

        if self.remove_matched_outputs:
            new_outputs_trimmed: list[SSAValue] = []
            output_types_trimmed: list[Attribute] = []
            for i in range(len(new_outputs)):
                if i not in matched_output_idxes:
                    new_outputs_trimmed.append(new_outputs[i])
                    output_types_trimmed.append(output_types[i])
            new_outputs, output_types = new_outputs_trimmed, output_types_trimmed

        func_op.function_type = FunctionType.from_lists(
            func_op.function_type.inputs.data, output_types
        )
        new_ops.append(ReturnOp(*new_outputs))
        rewriter.replace_matched_op(new_ops)


@dataclass(frozen=True)
class JaxUseDonatedArguments(ModulePass):
    name = "jax-use-donated-arguments"

    remove_matched_outputs: bool = False

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [SubstituteDonatedTensors(self.remove_matched_outputs)]
            ),
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=True,
        )
        the_one_pass.rewrite_module(op)
