from collections.abc import Sequence

from attr import dataclass

from xdsl.context import MLContext
from xdsl.dialects import memref, stencil, tensor
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.csl import csl_stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class ConvertAccessOpFromPrefetchPattern(RewritePattern):
    """
    Rebuilds stencil.access by csl_stencil.access which operates on prefetched accesses.

    stencil.access operates on stencil.temp types found at arg_index
    csl_stencil.access operates on memref< num_neighbors x tensor< buf_size x data_type >> found at last arg index

    Note: This is intended to be called in a nested pattern rewriter, such that the above precondition is met.
    """

    arg_index: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.AccessOp, rewriter: PatternRewriter, /):
        assert len(op.offset) == 2
        if (
            tuple(op.offset) == (0, 0)
            or op.temp != op.get_apply().region.block.args[self.arg_index]
        ):
            return
        prefetched_arg = op.get_apply().region.block.args[-1]
        assert isa(m_type := prefetched_arg.type, memref.MemRefType[Attribute])
        assert isa(t_type := m_type.get_element_type(), TensorType[Attribute])

        csl_access_op = csl_stencil.AccessOp(
            op=prefetched_arg,
            offset=op.offset,
            offset_mapping=op.offset_mapping,
            result_type=t_type,
        )

        # The stencil-tensorize-z-dimension pass inserts tensor.ExtractSliceOps after stencil.access to remove ghost cells.
        # Since ghost cells are not prefetched, these ops can be removed again. Check if the ExtractSliceOp
        # has no other effect and if so, remove both.
        if (
            len(op.res.uses) == 1
            and isinstance(use := list(op.res.uses)[0].operation, tensor.ExtractSliceOp)
            and tuple(d.data for d in use.static_sizes.data) == t_type.get_shape()
            and tuple(d.data for d in use.static_offsets.data) == (0,)
            and tuple(d.data for d in use.static_strides.data) == (1,)
            and len(use.offsets) == 0
            and len(use.sizes) == 0
            and len(use.strides) == 0
        ):
            rewriter.replace_op(use, csl_access_op)
            rewriter.erase_op(op)
        else:
            rewriter.replace_matched_op(csl_access_op)


@dataclass(frozen=True)
class ConvertSwapToPrefetchPattern(RewritePattern):
    """
    Translates dmp.swap to csl_stencil.prefetch
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        # remove op if it contains no swaps
        if op.swaps is None or len(op.swaps) == 0:
            rewriter.erase_matched_op(False)
            return

        assert all(
            len(swap.size) == 3 for swap in op.swaps
        ), "currently only 3-dimensional stencils are supported"
        assert all(
            swap.size[:2] == (1, 1) for swap in op.swaps
        ), "invoke dmp to decompose from (x,y,z) to (1,1,z)"
        # check that size is uniform
        uniform_size = op.swaps.data[0].size[2]
        assert all(
            swap.size[2] == uniform_size for swap in op.swaps
        ), "all swaps need to be of uniform size"

        assert isinstance(op.input_stencil, OpResult)
        assert isa(
            op.input_stencil.type,
            memref.MemRefType[Attribute] | stencil.TempType[Attribute],
        )
        assert isa(
            t_type := op.input_stencil.type.get_element_type(), TensorType[Attribute]
        )

        # when translating swaps, remove third dimension
        prefetch_op = csl_stencil.PrefetchOp(
            input_stencil=op.input_stencil.op,
            topo=op.topo,
            swaps=[
                csl_stencil.ExchangeDeclarationAttr(swap.neighbor[:2])
                for swap in op.swaps
            ],
            result_type=memref.MemRefType(
                TensorType(t_type.get_element_type(), (uniform_size,)),
                (len(op.swaps),),
            ),
        )

        # a little hack to get around a check that prevents replacing a no-results op with an n-results op
        rewriter.replace_matched_op(prefetch_op, new_results=[])

        # uses have to be retrieved *before* the loop because of the rewriting happening inside the loop
        uses = list(op.input_stencil.uses)

        # csl_stencil.prefetch, unlike dmp.swap, has a return value. This is added as the last arg
        # to stencil.apply, before rebuilding the op and replacing stencil.access ops by csl_stencil.access ops
        # that reference the prefetched buffers (note, this is only done for neighbor accesses)
        for use in uses:
            if not isinstance(use.operation, stencil.ApplyOp):
                continue
            apply_op = use.operation

            # arg_idx points to the stencil.temp type whose data is prefetched in a separate buffer
            arg_idx = apply_op.args.index(op.input_stencil)

            # add the prefetched buffer as the last arg to stencil.access
            apply_op.region.block.insert_arg(
                prefetch_op.result.type, len(apply_op.args)
            )

            # rebuild stencil.apply op
            r_types = [r.type for r in apply_op.results]
            assert isa(r_types, Sequence[stencil.TempType[Attribute]])
            new_apply_op = stencil.ApplyOp.get(
                [*apply_op.args, prefetch_op.result],
                apply_op.region.clone(),
                r_types,
            )
            rewriter.replace_op(apply_op, new_apply_op)

            # replace stencil.access (operating on stencil.temp at arg_index)
            # with csl_stencil.access (operating on memref at last arg index)
            nested_rewriter = PatternRewriteWalker(
                ConvertAccessOpFromPrefetchPattern(arg_idx)
            )

            nested_rewriter.rewrite_op(new_apply_op)


@dataclass(frozen=True)
class StencilToCslStencilPass(ModulePass):
    name = "stencil-to-csl-stencil"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(ConvertSwapToPrefetchPattern())
        module_pass.rewrite_module(op)
