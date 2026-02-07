import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.bufferization import MaterializeInDestinationOp
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FunctionType, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def map_donated_input_by_output(
    donatable_inputs: Sequence[BlockArgument], outputs: Sequence[SSAValue]
) -> dict[SSAValue, BlockArgument]:
    """
    Find suitable donated buffers for each of returned variables.
    Each buffer can be used only once.
    Types of the buffer and the variable should match.
    """

    donatable_inputs_by_type: dict[Attribute, list[BlockArgument]] = defaultdict(list)
    for inp in donatable_inputs:
        donatable_inputs_by_type[inp.type].append(inp)

    outputs_by_type: dict[Attribute, list[SSAValue]] = defaultdict(list)
    for out in outputs:
        if isinstance(out.owner, MaterializeInDestinationOp) and isinstance(
            out.owner.dest, BlockArgument
        ):
            # output has already been buffered
            continue
        outputs_by_type[out.type].append(out)

    mapping_by_type = {
        k: tuple(zip(donatable_inputs_by_type[k], outputs_by_type[k]))
        for k in donatable_inputs_by_type.keys() & outputs_by_type.keys()
    }

    return {o: i for mapping in mapping_by_type.values() for (i, o) in mapping}


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

        donated_input_mask = tuple(
            isinstance(inp.type, TensorType) and "tf.aliasing_output" in attr.data
            for inp, attr in zip(func_op.args, func_op.arg_attrs, strict=True)
        )
        donated_inputs = tuple(itertools.compress(func_op.args, donated_input_mask))

        if not donated_inputs:
            return

        donated_input_by_output = map_donated_input_by_output(
            donated_inputs, op.arguments
        )
        if not donated_input_by_output:
            return

        ordered_buffered_outputs = tuple(
            arg for arg in op.arguments if arg in donated_input_by_output
        )
        new_ops: list[Operation] = [
            MaterializeInDestinationOp(
                operands=[out, donated_input_by_output[out]], result_types=[out.type]
            )
            for out in ordered_buffered_outputs
        ]
        new_output_mapping = {
            out: mater_ops.results[0]
            for out, mater_ops in zip(ordered_buffered_outputs, new_ops, strict=True)
        }
        new_outputs = tuple(new_output_mapping.get(out, out) for out in op.arguments)

        return_types = tuple(func_op.function_type.outputs.data)
        if self.remove_matched_outputs:
            kept_outputs_mask = tuple(
                out not in donated_input_by_output for out in op.arguments
            )
            return_types = list(itertools.compress(return_types, kept_outputs_mask))
            new_outputs = list(itertools.compress(new_outputs, kept_outputs_mask))

        new_ops.append(ReturnOp(*new_outputs))
        func_op.function_type = FunctionType.from_lists(
            func_op.function_type.inputs.data, return_types
        )
        rewriter.replace_op(op, new_ops)

        # remove the donation attribute to avoid their reuse if we run the pass multiple times on the same function
        used_donated_arguments = set(donated_input_by_output.values())
        new_input_attrs = [dict(attr.data) for attr in func_op.arg_attrs]

        for inp, new_attr in zip(func_op.args, new_input_attrs, strict=True):
            if inp in used_donated_arguments:
                del new_attr["tf.aliasing_output"]

        func_op.arg_attrs = ArrayAttr(
            [DictionaryAttr(attr) for attr in new_input_attrs]
        )


@dataclass(frozen=True)
class JaxUseDonatedArguments(ModulePass):
    name = "jax-use-donated-arguments"

    remove_matched_outputs: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            SubstituteDonatedTensors(self.remove_matched_outputs),
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=True,
        ).rewrite_module(op)
