from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter


def fold_fills_in_module(module_op: ModuleOp):
    fill_op_by_memref: dict[SSAValue, memref_stream.FillOp] = {}
    for op in module_op.walk():
        if isinstance(op, memref_stream.FillOp):
            if op.memref in fill_op_by_memref:
                # Two consecutive fills, erase first one, and replace with the new one
                Rewriter.erase_op(fill_op_by_memref[op.memref])
            fill_op_by_memref[op.memref] = op
            continue

        if isinstance(op, memref_stream.GenericOp):
            fill_ops = tuple(
                fill_op_by_memref.get(output, None) for output in op.outputs
            )
            indices = tuple(
                index for index, value in enumerate(fill_ops) if value is not None
            )
            if indices and op.is_imperfectly_nested:
                # There are values to rewrite, replace the operation
                init_indices = ArrayAttr(IntAttr(index) for index in indices)
                inits = tuple(
                    fill_op.value for fill_op in fill_ops if fill_op is not None
                )
                Rewriter.replace_op(
                    op,
                    memref_stream.GenericOp(
                        op.inputs,
                        op.outputs,
                        inits,
                        Rewriter.move_region_contents_to_new_regions(op.body),
                        op.indexing_maps,
                        op.iterator_types,
                        op.bounds,
                        init_indices,
                        op.doc,
                        op.library_call,
                    ),
                )
                for fill_op in set(value for value in fill_ops if value is not None):
                    Rewriter.erase_op(fill_op)

        for operand in op.operands:
            if operand in fill_op_by_memref:
                del fill_op_by_memref[operand]


@dataclass(frozen=True)
class MemRefStreamFoldFillPass(ModulePass):
    """
    Folds `memref_stream.fill` operations that run immediately before a
    `memref_stream.generic` operation into the init value.
    Assumes that none of the memrefs involved are aliased.
    """

    name = "memref-stream-fold-fill"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        fold_fills_in_module(op)
