import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.bufferization import MaterializeInDestinationOp
from xdsl.dialects.builtin import FunctionType, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.irdl import VarOperand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def map_outputs_to_inputs(
    donated_inputs: Sequence[BlockArgument], outputs: VarOperand
) -> dict[int, int]:
    """
    Find suitable donated buffers for each of returned variables.
    Each buffer can be used only once.
    Types of the buffer and the variable should match.
    """
    used_inputs_idx: set[int] = set()
    output_input_mapping: dict[int, int] = {}

    for outputs_idx, output in enumerate(outputs):
        if isinstance(output.owner, MaterializeInDestinationOp) and isinstance(
            output.owner.dest, BlockArgument
        ):
            # the return value is already bufferized in a donated buffer
            continue
        for input_idx, arg in enumerate(donated_inputs):
            if arg.type == output.type and input_idx not in used_inputs_idx:
                output_input_mapping[outputs_idx] = input_idx
                used_inputs_idx.add(input_idx)
                break

    return output_input_mapping


def construct_new_output_list(
    donated_inputs: Sequence[BlockArgument],
    outputs: VarOperand,
    output_input_mapping: dict[int, int],
) -> tuple[Sequence[SSAValue], list[Operation]]:
    """
    Create new SSA values of buffers with the needed content, they will be used in the ReturnOp.
    Also create operations to associate buffers with corresponding return values.
    """
    new_outputs: Sequence[SSAValue] = [o for o in outputs]
    materialize_ops: list[Operation] = []

    for output_idx, input_idx in sorted(output_input_mapping.items()):
        materialize_ops.append(
            MaterializeInDestinationOp(
                operands=[outputs[output_idx], donated_inputs[input_idx]],
                result_types=[outputs[output_idx].type],
            )
        )
        new_outputs[output_idx] = materialize_ops[-1].results[0]

    return new_outputs, materialize_ops


@dataclass
class SubstituteDonatedTensors(RewritePattern):
    """
    Looks at returned tensors and if they match donated argument tensors ask bufferization to use them as buffers.
    """

    remove_matched_outputs: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        func_op = op.parent_op()
        assert isinstance(func_op, FuncOp)

        if func_op.arg_attrs is None or len(func_op.body.blocks) > 1:
            return

        donated_inputs_idx: list[int]
        donated_inputs: Sequence[BlockArgument]
        donated_inputs_idx, donated_inputs = zip(
            *[
                (arg_idx, inp)
                for arg_idx, (inp, attr) in enumerate(
                    zip(func_op.args, func_op.arg_attrs, strict=True)
                )
                if isinstance(inp.type, TensorType)
                and "tf.aliasing_output" in attr.data
            ]
        )

        if not donated_inputs:
            return

        output_input_mapping = map_outputs_to_inputs(donated_inputs, op.arguments)
        if not output_input_mapping:
            return

        new_outputs, new_ops = construct_new_output_list(
            donated_inputs, op.arguments, output_input_mapping
        )

        return_types = tuple(func_op.function_type.outputs.data)
        if self.remove_matched_outputs:
            kept_outputs_mask = tuple(
                i not in output_input_mapping for i in range(len(op.arguments))
            )
            return_types = list(itertools.compress(return_types, kept_outputs_mask))
            new_outputs = list(itertools.compress(new_outputs, kept_outputs_mask))

        new_ops.append(ReturnOp(*new_outputs))
        func_op.function_type = FunctionType.from_lists(
            func_op.function_type.inputs.data, return_types
        )
        rewriter.replace_matched_op(new_ops)

        # remove the donation attribute to avoid their reuse if we run the pass multiple times on the same function
        for input_idx in output_input_mapping.values():
            del func_op.arg_attrs.data[donated_inputs_idx[input_idx]].data[
                "tf.aliasing_output"
            ]


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
