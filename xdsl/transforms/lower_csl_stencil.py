from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
    i16,
)
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def get_dir_and_distance_ops(
    op: csl_stencil.AccessOp,
) -> tuple[Operation | None, Operation]:
    """
    Given an access op, return the distance and direction, assuming as access
    to a neighbour (not self) in a star-shape pattern
    """

    offset = tuple(op.offset)
    assert len(offset) == 2, "Expecting 2-dimensional access"
    assert (offset[0] == 0) ^ (
        offset[1] == 0
    ), "Expecting neighbour access in a star-shape pattern"
    # todo implement csl.direction
    # if offset[0] < 0:
    #     dir = "EAST"
    # elif offset[0] > 0:
    #     dir = "WEST"
    # elif offset[1] < 0:
    #     dir = "NORTH"
    # elif offset[1] > 0:
    #     dir = "SOUTH"
    # else:
    #     raise ValueError("Invalid offset, expecting 2-dimensional star-shape neighbor access")
    max_distance = abs(max(offset, key=abs))
    return None, arith.Constant(IntegerAttr(max_distance, 16))


def _get_module_wrapper(op: Operation) -> csl_wrapper.ModuleOp | None:
    """
    Return the enclosing csl_wrapper.module
    """
    parent_op = op.parent_op()
    while parent_op:
        if isinstance(parent_op, csl_wrapper.ModuleOp):
            return parent_op
        parent_op = parent_op.parent_op()
    return None


@dataclass(frozen=True)
class LowerAccessOp(RewritePattern):
    """
    Replaces `csl_stencil.access` with API calls.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.AccessOp, rewriter: PatternRewriter, /):
        if not (module_wrapper_op := _get_module_wrapper(op)):
            return

        # todo `dir` param:
        _, neighbor_op = get_dir_and_distance_ops(op)
        rewriter.replace_matched_op(
            [
                neighbor_op,
                m_call := csl.MemberCallOp(
                    "getRecvBufDsdByNeighbor",
                    csl.DsdType(csl.DsdKind.mem1d_dsd),
                    module_wrapper_op.get_program_import("stencil_comms.csl"),
                    [
                        # dir_op,
                        neighbor_op,
                    ],
                ),
                UnrealizedConversionCastOp.get([m_call], op.result_types),
            ]
        )


@dataclass(frozen=True)
class LowerApplyOp(RewritePattern):
    """
    Lowers csl_stencil.apply to an API call. Places the two regions in csl.funcs and
    passes them as callbacks.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        if not (module_wrapper_op := _get_module_wrapper(op)):
            return

        parent_func = op.parent_op()
        while parent_func:
            if isinstance(parent_func, func.FuncOp) or isinstance(
                parent_func, csl.FuncOp
            ):
                break
            parent_func = op.parent_op()
        assert (
            parent_func
        ), "Expected csl_stencil.apply to be inside a func.func or csl.func"

        cr = csl.FuncOp("chunk_reduce_cb", FunctionType.from_lists([i16], []))
        pp = csl.FuncOp(
            "post_process_cb", FunctionType.from_lists([], []), Region(Block())
        )
        cr.body.block.add_op(
            index_op := arith.IndexCastOp(
                cr.body.block.args[0],
                IndexType(),
            )
        )
        cr_arg_m = [
            op.communicated_stencil,  # buffer - this is a placeholder and should not be used after lowering AccessOp
            index_op.result,
            op.iter_arg,
            *op.args[: len(op.chunk_reduce.block.args) - 3],
        ]
        pp_arg_m = [op.communicated_stencil, op.iter_arg, *op.args[len(cr_arg_m) - 3 :]]

        rewriter.inline_block(
            op.chunk_reduce.block, InsertPoint.at_end(cr.body.block), cr_arg_m
        )
        rewriter.inline_block(
            op.post_process.block, InsertPoint.at_end(pp.body.block), pp_arg_m
        )

        rewriter.insert_op([cr, pp], InsertPoint.after(parent_func))

        num_chunks = arith.Constant(IntegerAttr(op.num_chunks.value, i16))
        api_call = csl.MemberCallOp(
            "communicate",
            None,
            module_wrapper_op.get_program_import("stencil_comms.csl"),
            [
                op.communicated_stencil,
                num_chunks,
                # todo cr function pointer
                # todo pp function pointer
            ],
        )

        rewriter.replace_matched_op([num_chunks, api_call], [])


@dataclass(frozen=True)
class LowerYieldOp(RewritePattern):
    """
    Lowers csl_stencil.yield to csl.return.
    Note, the callbacks generated return no values, whereas the yield op
    to be replaced may still report to yield values.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.YieldOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(csl.ReturnOp())


@dataclass(frozen=True)
class LowerCslStencil(ModulePass):
    """
    Lowers csl_stencil ops to csl and api calls.
    """

    name = "lower-csl-stencil"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAccessOp(),
                    LowerApplyOp(),
                    LowerYieldOp(),
                ]
            )
        )
        module_pass.rewrite_module(op)
