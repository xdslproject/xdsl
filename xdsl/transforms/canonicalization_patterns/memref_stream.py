from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveUnusedInitOperandPattern(RewritePattern):
    """
    Removes the inputs corresponding to unused arguments in the body.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if memref_stream.IteratorTypeAttr.interleaved() in op.iterator_types.data:
            # Do not run on interleaved ops
            return

        block = op.body.block
        block_args = block.args

        inputs = op.inputs

        num_inputs = len(inputs)

        unused_input_indices = tuple(
            index for index, arg in enumerate(block_args[:num_inputs]) if not arg.uses
        )

        if not unused_input_indices:
            # All args have uses, nothing to remove
            return

        outputs = op.outputs

        optional_inits: list[SSAValue | None] = [None] * len(outputs)
        for init_index, init in zip(op.init_indices, op.inits, strict=True):
            optional_inits[init_index.data] = init

        assert len(optional_inits) == len(outputs)

        new_inputs: list[SSAValue] = []
        new_indexing_maps: list[AffineMapAttr] = []

        for index, arg in enumerate(block_args[:num_inputs]):
            drop_operand = index in unused_input_indices
            if drop_operand:
                arg.erase()
                continue

            new_indexing_maps.append(op.indexing_maps.data[index])
            new_inputs.append(inputs[index])

        new_indexing_maps.extend(op.indexing_maps.data[num_inputs:])

        for i in reversed(unused_input_indices):
            block.erase_arg(block_args[i])

        rewriter.replace_op(
            op,
            memref_stream.GenericOp(
                new_inputs,
                op.outputs,
                op.inits,
                rewriter.move_region_contents_to_new_regions(op.body),
                ArrayAttr(new_indexing_maps),
                op.iterator_types,
                op.bounds,
                op.init_indices,
                op.doc,
                op.library_call,
            ),
        )
