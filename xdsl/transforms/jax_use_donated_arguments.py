import itertools
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.bufferization import MaterializeInDestinationOp
from xdsl.dialects.builtin import FunctionType, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue
from xdsl.irdl import VarOperand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def map_outputs_to_inputs(donated_inputs: list[BlockArgument], outputs: VarOperand):
    used_inputs_idx: set[int] = set()
    output_input_mapping: list[int | None] = [None for _ in range(len(outputs))]

    for outputs_idx, output in enumerate(outputs):
        for input_idx, arg in enumerate(donated_inputs):
            if arg.type == output.type and input_idx not in used_inputs_idx:
                output_input_mapping[outputs_idx] = input_idx
                used_inputs_idx.add(input_idx)
                break

    return output_input_mapping


def construct_new_output_list(
    donated_inputs: list[BlockArgument],
    outputs: VarOperand,
    output_input_mapping: list[int | None],
):
    new_outputs: list[SSAValue] = []
    materialize_ops: list[Operation] = []

    for output_idx, input_idx in enumerate(output_input_mapping):
        if input_idx is None:
            new_outputs.append(outputs[output_idx])
        else:
            materialize_ops.append(
                MaterializeInDestinationOp(
                    operands=[outputs[output_idx], donated_inputs[input_idx]],
                    result_types=[outputs[output_idx].type],
                )
            )
            new_outputs.append(materialize_ops[-1].results[0])

    return new_outputs, materialize_ops


def construct_return(
    output_list: list[SSAValue],
    output_types: list[Attribute],
    output_input_mapping: list[int | None],
    remove_matched_outputs: bool,
):
    if remove_matched_outputs:
        kept_outputs_mask = [i is None for i in output_input_mapping]
        output_types = list(itertools.compress(output_types, kept_outputs_mask))
        output_list = list(itertools.compress(output_list, kept_outputs_mask))

    return ReturnOp(*output_list), output_types


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

        if not donated_inputs:
            return

        output_input_mapping = map_outputs_to_inputs(donated_inputs, op.arguments)
        if all(map(lambda x: x is None, output_input_mapping)):
            return

        new_outputs, new_ops = construct_new_output_list(
            donated_inputs, op.arguments, output_input_mapping
        )
        return_op, return_types = construct_return(
            new_outputs,
            list(func_op.function_type.outputs.data),
            output_input_mapping,
            self.remove_matched_outputs,
        )
        new_ops.append(return_op)
        func_op.function_type = FunctionType.from_lists(
            func_op.function_type.inputs.data, return_types
        )
        rewriter.replace_matched_op(new_ops)


@dataclass(frozen=True)
class JaxUseDonatedArguments(ModulePass):
    name = "jax-use-donated-arguments"

    remove_matched_outputs: bool = False

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            SubstituteDonatedTensors(self.remove_matched_outputs),
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=True,
        ).rewrite_module(op)
