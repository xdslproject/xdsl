from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import memref, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.csl import csl_stencil, csl_wrapper
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class MaterializeInApplyDest(RewritePattern):
    """
    Stores the yielded values to the buffers specified in `apply.dest` instead of yielding them.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.YieldOp, rewriter: PatternRewriter, /):
        if not len(op.arguments) > 0:
            return
        assert isinstance(apply := op.parent_op(), csl_stencil.ApplyOp)

        if op.parent_region() != apply.done_exchange:
            return

        views: list[Operation] = []
        add_args: list[SSAValue] = []
        for src, dst in zip(op.arguments, apply.dest, strict=True):
            assert isa(src.type, memref.MemRefType)
            assert isa(dst.type, memref.MemRefType)
            dst_arg = apply.done_exchange.block.insert_arg(
                dst.type, len(apply.done_exchange.block.args)
            )
            views.append(
                memref.SubviewOp.get(
                    dst_arg,
                    [
                        (d - s) // 2  # symmetric offset
                        for s, d in zip(src.type.get_shape(), dst.type.get_shape())
                    ],
                    src.type.get_shape(),
                    len(src.type.get_shape()) * [1],
                    src.type,
                )
            )
            add_args.append(dst)
        copies = [memref.CopyOp(src, dst) for src, dst in zip(op.arguments, views)]
        rewriter.insert_op(
            [*views, *copies],
            InsertPoint.before(op),
        )

        rewriter.replace_op(op, csl_stencil.YieldOp())
        rewriter.replace_op(
            apply,
            csl_stencil.ApplyOp(
                operands=[
                    apply.field,
                    apply.accumulator,
                    apply.args_rchunk,
                    [*apply.args_dexchng, *add_args],
                    apply.dest,
                ],
                regions=[apply.detach_region(r) for r in apply.regions],
                properties=apply.properties,
                result_types=apply.result_types or [[]],
            ),
        )


@dataclass(frozen=True)
class DisableComputeInBorderRegion(RewritePattern):
    """
    Processing elements in the border region do not need to do compute or store their values back to a buffer.
    For simplicity, wrap the full `csl_stencil.apply.done_exchange` region in an `scf.if`.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        wrapper_op = op.parent_op()
        while wrapper_op and not isinstance(wrapper_op, csl_wrapper.ModuleOp):
            wrapper_op = wrapper_op.parent_op()
        if not wrapper_op:
            return

        cond = wrapper_op.get_program_param("isBorderRegionPE")
        if cond in op.args_dexchng:
            return

        op.done_exchange.block.insert_arg(cond.type, len(op.done_exchange.block.args))

        rewriter.insert_op(
            if_op := scf.IfOp(
                op.done_exchange.block.args[-1], [], Region(Block()), Region(Block())
            ),
            InsertPoint.at_start(op.done_exchange.block),
        )

        assert if_op.next_op, "Block cannot be empty"

        if (
            not isinstance(yld := op.done_exchange.block.last_op, csl_stencil.YieldOp)
            or len(yld.arguments) > 0
        ):
            return

        body = op.done_exchange.block.split_before(if_op.next_op)
        rewriter.inline_block(body, InsertPoint.at_start(if_op.false_region.block))

        rewriter.insert_op(
            csl_stencil.YieldOp(), InsertPoint.at_end(op.done_exchange.block)
        )
        rewriter.replace_op(yld, scf.YieldOp())
        rewriter.insert_op(scf.YieldOp(), InsertPoint.at_start(if_op.true_region.block))
        rewriter.replace_op(
            op,
            csl_stencil.ApplyOp(
                operands=[
                    op.field,
                    op.accumulator,
                    op.args_rchunk,
                    [*op.args_dexchng, cond],
                    op.dest,
                ],
                regions=[op.detach_region(r) for r in op.regions],
                properties=op.properties,
                result_types=op.result_types or [[]],
            ),
        )


@dataclass(frozen=True)
class CslStencilMaterializeStores(ModulePass):
    """
    This pass creates stores for values yielded from `csl_stencil.apply.done_exchange.yield`
    to the buffers in `apply.dest`.
    Stores should only be materialised for PEs not in the border region.

    The pass operates on memrefs, run after bufferization.
    """

    name = "csl-stencil-materialize-stores"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MaterializeInApplyDest(),
                    DisableComputeInBorderRegion(),
                ]
            ),
            walk_regions_first=True,
        ).rewrite_module(op)
