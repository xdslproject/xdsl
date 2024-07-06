from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntAttr
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveUnusedOperandPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if memref_stream.IteratorTypeAttr.interleaved() in op.iterator_types.data:
            # Do not run on interleaved ops
            return

        block = op.body.block
        block_args = block.args

        unused_indices = tuple(
            index for index, arg in enumerate(block_args) if not arg.uses
        )

        if not unused_indices:
            # All args have uses, nothing to remove
            return

        inputs = op.inputs
        outputs = op.outputs

        num_inputs = len(inputs)

        optional_inits: list[SSAValue | None] = [None] * len(outputs)
        for init_index, init in zip(op.init_indices, op.inits, strict=True):
            optional_inits[init_index.data] = init

        assert len(optional_inits) == len(outputs)

        new_inputs: list[SSAValue] = []
        new_outputs: list[SSAValue] = []
        new_indexing_maps: list[AffineMapAttr] = []
        new_optional_inits: list[SSAValue | None] = []

        num_outputs_dropped = 0
        for index, arg in enumerate(block_args):
            drop_operand = index in unused_indices
            is_input = index <= num_inputs
            if drop_operand:
                num_outputs_dropped += not is_input
                arg.erase()
                continue

            new_indexing_maps.append(op.indexing_maps.data[index])
            if is_input:
                new_inputs.append(inputs[index])
            else:
                output_index = index - num_inputs
                new_optional_inits.append(optional_inits[output_index])
                new_outputs.append(outputs[output_index])

        assert len(new_outputs) == len(new_optional_inits)

        new_inits: list[SSAValue] = []
        new_init_indices: list[IntAttr] = []

        for index, init in enumerate(new_optional_inits):
            if init is not None:
                new_inits.append(init)
                new_init_indices.append(IntAttr(index))

        for arg in reversed(block_args):
            if not arg.uses:
                block.erase_arg(arg)

        rewriter.replace_matched_op(
            memref_stream.GenericOp(
                new_inputs,
                new_outputs,
                new_inits,
                rewriter.move_region_contents_to_new_regions(op.body),
                ArrayAttr(new_indexing_maps),
                op.iterator_types,
                op.bounds,
                ArrayAttr(new_init_indices),
            )
        )
