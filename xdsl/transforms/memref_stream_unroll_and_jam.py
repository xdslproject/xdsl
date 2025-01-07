from dataclasses import dataclass
from itertools import repeat

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, ModuleOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.ir.affine import AffineExpr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import DiagnosticException


def unroll_and_jam(
    op: memref_stream.GenericOp,
    rewriter: PatternRewriter | Rewriter,
    iterator_index: int,
    unroll_factor: int,
):
    if unroll_factor == 1:
        # If unroll factor is 1, rewrite is a no-op
        return

    old_block = op.body.block
    new_region = Region(
        Block(
            arg_types=(
                t for arg in old_block.args for t in repeat(arg.type, unroll_factor)
            )
        )
    )
    with ImplicitBuilder(new_region) as args:
        # For each interleaved block replica, a mapping from old values to new values
        value_map: tuple[dict[SSAValue, SSAValue], ...] = tuple(
            {} for _ in range(unroll_factor)
        )
        for arg_index, new_arg in enumerate(args):
            old_arg = old_block.args[arg_index // unroll_factor]
            value_map[arg_index % unroll_factor][old_arg] = new_arg
            new_arg.name_hint = old_arg.name_hint
        for block_op in old_block.ops:
            if isinstance(block_op, memref_stream.YieldOp):
                memref_stream.YieldOp(
                    *([vm[arg] for vm in value_map for arg in block_op.arguments])
                )
            else:
                for i in range(unroll_factor):
                    block_op.clone(value_mapper=value_map[i])

    # New maps are the same, except that they have one more dimension and the
    # dimension that is interleaved is updated to
    # `dim * interleave_factor + new_dim`.
    new_indexing_maps = ArrayAttr(
        AffineMapAttr(
            m.data.replace_dims_and_symbols(
                (
                    tuple(AffineExpr.dimension(i) for i in range(iterator_index))
                    + (
                        AffineExpr.dimension(iterator_index) * unroll_factor
                        + AffineExpr.dimension(m.data.num_dims),
                    )
                    + tuple(
                        AffineExpr.dimension(i)
                        for i in range(iterator_index + 1, m.data.num_dims + 2)
                    )
                ),
                (),
                m.data.num_dims + 1,
                0,
            )
        )
        for m in op.indexing_maps
    )

    # The new bounds are the same, except there is one more bound
    new_bounds = list(op.bounds)
    new_bounds.append(IntegerAttr.from_index_int_value(unroll_factor))
    iterator_ub = op.bounds.data[iterator_index].value.data
    new_bounds[iterator_index] = IntegerAttr.from_index_int_value(
        iterator_ub // unroll_factor
    )

    rewriter.replace_op(
        op,
        memref_stream.GenericOp(
            op.inputs,
            op.outputs,
            op.inits,
            new_region,
            new_indexing_maps,
            ArrayAttr(
                op.iterator_types.data + (memref_stream.IteratorTypeAttr.interleaved(),)
            ),
            ArrayAttr(new_bounds),
            op.init_indices,
            op.doc,
            op.library_call,
        ),
    )


@dataclass(frozen=True)
class MemrefStreamUnrollAndJamPass(ModulePass):
    """
    TODO
    """

    name = "memref-stream-unroll-and-jam"

    op_index: int
    iterator_index: int
    unroll_factor: int

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        msg_ops = (
            child for child in op.walk() if isinstance(child, memref_stream.GenericOp)
        )

        msg_op = None

        for index, child in enumerate(msg_ops):
            if index == self.op_index:
                msg_op = child
                break

        if msg_op is None:
            raise DiagnosticException("Index out of bounds")

        unroll_and_jam(msg_op, Rewriter(), self.iterator_index, self.unroll_factor)
