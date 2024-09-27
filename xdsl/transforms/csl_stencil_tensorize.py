from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from xdsl.context import MLContext
from xdsl.dialects import stencil, tensor
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, TensorType
from xdsl.dialects.csl import csl_stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


def get_tile_size(bounds: stencil.StencilBoundsAttr | None) -> tuple[int, ...]:
    if bounds:
        return tuple(b for b in bounds.ub if b > 1)
    return ()


# def get_active_dims(bounds: stencil.StencilBoundsAttr | None):
#     if bounds:
#         return tuple(i for i, v in enumerate(bounds.ub) if v > 1)
#     return ()


_T = TypeVar("_T")


def filter_active_dims(
    bounds: stencil.StencilBoundsAttr | None, s: Iterable[_T]
) -> Sequence[_T]:
    if bounds:
        return tuple(v for b, v in zip(bounds.ub, s) if b > 1)
    return ()


def filter_inactive_dims(
    bounds: stencil.StencilBoundsAttr | None, s: Iterable[_T]
) -> Sequence[_T]:
    if bounds:
        return tuple(v for b, v in zip(bounds.ub, s) if b <= 1)
    return ()


def tensorize_stencil_type(
    tile: Sequence[int], field: stencil.FieldType[Attribute]
) -> stencil.FieldType[Attribute]:
    elem_t = field.get_element_type()
    if isa(elem_t, TensorType[Attribute]):
        return field

    assert isinstance(field.bounds, stencil.StencilBoundsAttr)

    bounds = list(zip(field.bounds.lb, field.bounds.ub))
    if len(bounds) > 2:
        typ = TensorType(elem_t, (bounds[2][1] - bounds[2][0],))
    else:
        typ = TensorType(elem_t, tile)

    return stencil.FieldType[Attribute](bounds[:2], typ)


class UpdateAccumulatorShape(RewritePattern):
    """
    Updates the accumulator shape:
      * of the accumulator operand
      * of the accumulator block args
      * of the types yielded back (with an insert_slice_op if communication is broken down into chunks)
      * of the receive_chunk neighbor buffer shape (which is len(swaps) times the accumulator shape)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        # check if shape needs updating
        if not isa(
            op.accumulator.type, TensorType[Attribute]
        ) or op.accumulator.type.get_shape() == (tile := get_tile_size(op.bounds)):
            return

        # setup new accumulator
        elem_t = op.accumulator.type.get_element_type()
        rewriter.insert_op(
            acc := tensor.EmptyOp(
                (),
                TensorType(elem_t, tile),
            ),
            InsertPoint.before(op),
        )

        # updating regions' accumulator block args
        assert len(tile) == 1, "Tensor tiles of >1 dims currently not implemented"
        if (tile[0] % op.num_chunks.value.data) != 0:
            raise ValueError("Tensor tile does not evenly divide by chunk size")
        chunk_size = tile[0] // op.num_chunks.value.data
        chunk_t = TensorType(elem_t, (chunk_size,))
        recv_chunk_acc = op.receive_chunk.block.args[2]
        done_exchange_acc = op.done_exchange.block.args[1]
        rewriter.modify_value_type(recv_chunk_acc, acc.tensor.type)
        rewriter.modify_value_type(done_exchange_acc, acc.tensor.type)

        # updating regions' yields
        assert isinstance(
            rc_yield := op.receive_chunk.block.last_op, csl_stencil.YieldOp
        )
        assert isinstance(
            de_yield := op.done_exchange.block.last_op, csl_stencil.YieldOp
        )
        assert len(rc_yield.operands) == 1, "Must yield exactly one value"
        assert len(de_yield.operands) == 1, "Must yield exactly one value"
        rewriter.modify_value_type(de_yield.operands[0], acc.tensor.type)
        if rc_yield.operands[0] != recv_chunk_acc:
            rewriter.modify_value_type(rc_yield.operands[0], chunk_t)

        # if communication is broken up into more than one chunk, insert chunk before yielding into the accumulator
        if op.num_chunks.value.data > 1:
            rewriter.replace_op(
                rc_yield,
                [
                    insert_slice_op := tensor.InsertSliceOp.get(
                        source=rc_yield.operands[0],
                        dest=recv_chunk_acc,
                        offsets=(op.receive_chunk.block.args[1],),
                        static_sizes=(chunk_size,),
                    ),
                    csl_stencil.YieldOp(insert_slice_op.result),
                ],
            )

        # update neighbor buf in chunk_receive region to n*chunks shape
        neighbor_buf = op.receive_chunk.block.args[0]
        if isinstance(neighbor_buf.type, stencil.StencilType):
            rewriter.modify_value_type(
                neighbor_buf, TensorType(elem_t, [len(op.swaps), *chunk_t.get_shape()])
            )

        # rebuild apply op
        rewriter.replace_matched_op(
            csl_stencil.ApplyOp(
                operands=[
                    op.field,
                    acc,
                    op.args,
                    op.dest,
                ],
                properties=op.properties.copy(),
                regions=[op.detach_region(r) for r in op.regions],
                result_types=[op.result_types],
            )
        )


# class UpdateRecvChunkNeighborShape(RewritePattern):
#     """Tensorize accessed neighbour field in apply.receive_chunk's"""
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
#         # check if shape needs updating
#         if not isinstance(main_buf := op.receive_chunk.block.args[0], stencil.StencilType) or not isa(
#             t := op.accumulator.type, TensorType[Attribute]
#         ):
#             return
#         rewriter.modify_value_type(main_buf, TensorType(t.get_element_type(), [len(op.swaps), *t.get_shape()]))


class SetApplyOffsets(RewritePattern):
    """Set apply offsets for the dimensions to be tensorised."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        if (
            op.offsets
            or not isa(op.field.type, stencil.StencilType[Attribute])
            or not isinstance(op.field.type.bounds, stencil.StencilBoundsAttr)
            or not isinstance(op.bounds, stencil.StencilBoundsAttr)
        ):
            return
        offsets = stencil.IndexAttr.get(
            *(
                lb - fld
                for lb, fld in filter_active_dims(
                    op.bounds, zip(op.bounds.lb, op.field.type.bounds.lb)
                )
            )
        )
        rewriter.replace_matched_op(
            csl_stencil.ApplyOp(
                operands=[op.field, op.accumulator, op.args, op.dest],
                result_types=[op.result_types],
                regions=[op.detach_region(r) for r in op.regions],
                properties={**op.properties, "offsets": offsets},
            )
        )


class TensorizeApplySwaps(RewritePattern):
    """Translate apply.swaps from `[dmp.ExchangeDeclarationAttr]` to `[csl_stencil.ExchangeDeclarationAttr]`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        if not isa(op.swaps, ArrayAttr[dmp.ExchangeDeclarationAttr]) or not isinstance(
            op.bounds, stencil.StencilBoundsAttr
        ):
            return

        # rebuild swaps
        swaps = ArrayAttr(
            csl_stencil.ExchangeDeclarationAttr(
                filter_inactive_dims(op.bounds, swap.neighbor)
            )
            for swap in op.swaps
        )
        rewriter.replace_matched_op(
            csl_stencil.ApplyOp(
                operands=[op.field, op.accumulator, op.args, op.dest],
                result_types=[op.result_types],
                regions=[op.detach_region(r) for r in op.regions],
                properties={**op.properties, "swaps": swaps},
            )
        )


@dataclass
class StencilTypeConversion(TypeConversionPattern):
    tile_size: tuple[int, ...] = ()

    @attr_type_rewrite_pattern
    def convert_type(
        self, field: stencil.FieldType[Attribute]
    ) -> stencil.FieldType[Attribute]:
        return tensorize_stencil_type(self.tile_size, field)


@dataclass(frozen=True)
class CslStencilTensorize(ModulePass):
    name = "csl-stencil-tensorize"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        tile_sizes = set[tuple[int, ...]]()
        for apply_op in op.walk():
            if isinstance(apply_op, csl_stencil.ApplyOp):
                tile_sizes.add(get_tile_size(apply_op.bounds))
        if len(tile_sizes) != 1 or not (tile_size := tile_sizes.pop()):
            raise ValueError("Encountered multiple tile sizes, cannot apply pass")

        prepass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    UpdateAccumulatorShape(),
                    # UpdateRecvChunkNeighborShape(),
                    SetApplyOffsets(),
                    TensorizeApplySwaps(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=True,
        )
        prepass.rewrite_module(op)
        main_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # FuncOpTensorize(),
                    StencilTypeConversion(
                        tile_size=tile_size
                    ),  # this needs to come after FuncOpTensorize()
                    # LoadOpTensorize(),
                    # ApplyOpTensorize(),
                    # StoreOpTensorize(),
                    # DmpSwapOpTensorize(),
                    # AccessOpTensorize(),   # this doesn't work here, using second pass
                ]
            ),
            walk_reverse=False,
            apply_recursively=True,
        )
        main_pass.rewrite_module(op)
