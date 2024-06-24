from collections.abc import Sequence

from attr import dataclass

from xdsl.context import MLContext
from xdsl.dialects import memref, stencil, tensor
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, TensorType
from xdsl.dialects.csl_ import csl_stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class AccessOpFromPrefetch(RewritePattern):

    # index of the stencil access to be replaced by the prefetched buffer in the last arg
    arg_index: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.AccessOp, rewriter: PatternRewriter, /):
        assert len(op.offset) == 2
        if tuple(op.offset) == (0, 0):
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

        # if the matched op is immediately followed by tensor.ExtractSliceOps to remove ghost cells, replace both
        if len(op.res.uses) == 1 and isinstance(
            use := list(op.res.uses)[0].operation, tensor.ExtractSliceOp
        ):
            rewriter.replace_op(use, csl_access_op)
            rewriter.erase_op(op)
        else:
            rewriter.replace_matched_op(csl_access_op)


@dataclass(frozen=True)
class SwapToPrefetch(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        if op.swaps is None or len(op.swaps) == 0:
            rewriter.erase_matched_op(False)
            return
        assert isinstance(op.input_stencil, OpResult)
        # currently only works for 3-dimensional stencils
        assert all(len(swap.size) == 3 for swap in op.swaps)
        # dmp should decompose from (x,y,z) to (1,1,z)
        assert all(swap.size[:2] == (1, 1) for swap in op.swaps)
        # check that size is uniform
        uniform_size = op.swaps.data[0].size[2]
        assert all(swap.size[2] == uniform_size for swap in op.swaps)

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
            size=IntegerAttr(uniform_size, IntegerType(64)),
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

        # rewrite stencil.apply
        for use in uses:
            if not isinstance(use.operation, stencil.ApplyOp):
                continue
            apply_op = use.operation
            arg_idx = apply_op.args.index(op.input_stencil)

            apply_op.region.block.insert_arg(
                prefetch_op.result.type, len(apply_op.args)
            )
            r_types = [r.type for r in apply_op.results]
            assert isa(r_types, Sequence[stencil.TempType[Attribute]])
            new_apply_op = stencil.ApplyOp.get(
                list(apply_op.args) + [prefetch_op.result],
                apply_op.region.clone(),
                r_types,
            )
            rewriter.replace_op(apply_op, new_apply_op)

            nested_rewriter = PatternRewriteWalker(
                GreedyRewritePatternApplier(
                    [
                        AccessOpFromPrefetch(arg_idx),
                    ]
                ),
                walk_reverse=False,
                apply_recursively=False,
            )

            nested_rewriter.rewrite_op(new_apply_op)


@dataclass(frozen=True)
class StencilToCslStencilPass(ModulePass):
    name = "stencil-to-csl-stencil"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([SwapToPrefetch()]),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
