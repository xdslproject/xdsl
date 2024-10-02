from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, func, memref
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
    i16,
)
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
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


def get_dir_and_distance_ops(
    op: csl_stencil.AccessOp,
) -> tuple[csl.DirectionOp, arith.Constant]:
    """
    Given an access op, return the distance and direction, assuming as access
    to a neighbour (not self) in a star-shape pattern
    """

    offset = tuple(op.offset)
    assert len(offset) == 2, "Expecting 2-dimensional access"
    assert (offset[0] == 0) != (
        offset[1] == 0
    ), "Expecting neighbour access in a star-shape pattern"
    if offset[0] < 0:
        d = csl.Direction.EAST
    elif offset[0] > 0:
        d = csl.Direction.WEST
    elif offset[1] < 0:
        d = csl.Direction.NORTH
    elif offset[1] > 0:
        d = csl.Direction.SOUTH
    else:
        raise ValueError(
            "Invalid offset, expecting 2-dimensional star-shape neighbor access"
        )
    max_distance = abs(max(offset, key=abs))
    return csl.DirectionOp(d), arith.Constant(IntegerAttr(max_distance, 16))


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

        dir_op, neighbor_op = get_dir_and_distance_ops(op)
        rewriter.replace_matched_op(
            [
                neighbor_op,
                dir_op,
                m_call := csl.MemberCallOp(
                    "getRecvBufDsdByNeighbor",
                    csl.DsdType(csl.DsdKind.mem1d_dsd),
                    module_wrapper_op.get_program_import("stencil_comms.csl"),
                    [
                        dir_op,
                        neighbor_op,
                    ],
                ),
                UnrealizedConversionCastOp.get([m_call], op.result_types),
            ]
        )


@dataclass
class LowerApplyOp(RewritePattern):
    """
    Lowers csl_stencil.apply to an API call. Places the two regions in csl.funcs and
    passes them as callbacks.
    """

    count: int = 0

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

        # set up csl funcs
        chunk_fn = csl.FuncOp(
            "receive_chunk_cb" + str(self.count), FunctionType.from_lists([i16], [])
        )
        chunk_fn.body.block.args[0].name_hint = "offset"
        done_fn = csl.FuncOp(
            "done_exchange_cb" + str(self.count),
            FunctionType.from_lists([], []),
            Region(Block()),
        )
        self.count += 1

        # the offset arg was of type index and is now i16, so it's cast back to index to be used in the func body
        chunk_fn.body.block.add_op(
            index_op := arith.IndexCastOp(
                chunk_fn.body.block.args[0],
                IndexType(),
            )
        )

        # arg maps for the regions
        chunk_arg_m = [
            op.field,  # buffer - this is a placeholder and should not be used after lowering AccessOp
            index_op.result,
            op.accumulator,
            *op.args[: len(op.receive_chunk.block.args) - 3],
        ]
        done_arg_m = [
            op.field,
            op.accumulator,
            *op.args[len(chunk_arg_m) - 3 :],
        ]
        index_op.result.name_hint = "offset"
        op.accumulator.name_hint = "accumulator"

        # inlining both regions
        rewriter.inline_block(
            op.receive_chunk.block,
            InsertPoint.at_end(chunk_fn.body.block),
            chunk_arg_m,
        )
        rewriter.inline_block(
            op.done_exchange.block, InsertPoint.at_end(done_fn.body.block), done_arg_m
        )

        # place both func next to the enclosing parent func
        rewriter.insert_op([chunk_fn, done_fn], InsertPoint.after(parent_func))

        # ensure we send only core data
        assert isa(op.accumulator.type, memref.MemRefType[Attribute])
        assert isa(op.field.type, memref.MemRefType[Attribute])
        send_buf = memref.Subview.get(
            op.field,
            [
                (d - s) // 2  # symmetric offset
                for s, d in zip(
                    op.accumulator.type.get_shape(), op.field.type.get_shape()
                )
            ],
            op.accumulator.type.get_shape(),
            len(op.accumulator.type.get_shape()) * [1],
            op.accumulator.type,
        )

        # add api call
        num_chunks = arith.Constant(IntegerAttr(op.num_chunks.value, i16))
        chunk_ref = csl.AddressOfFnOp(chunk_fn)
        done_ref = csl.AddressOfFnOp(done_fn)
        # send_buf = memref.Subview.get(op.field, [], op.accumulator.type.get_shape(), )
        api_call = csl.MemberCallOp(
            "communicate",
            None,
            module_wrapper_op.get_program_import("stencil_comms.csl"),
            [
                send_buf,
                num_chunks,
                chunk_ref,
                done_ref,
            ],
        )

        # replace op with api call
        rewriter.replace_matched_op(
            [num_chunks, chunk_ref, done_ref, send_buf, api_call], []
        )


@dataclass(frozen=True)
class LowerYieldOp(RewritePattern):
    """
    Lowers csl_stencil.yield to csl.return.
    Note, the callbacks generated return no values, and the yield op
    to be replaced should also yield no values. This should be run
    after `--csl-stencil-materialize-stores`.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.YieldOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(csl.ReturnOp())


@dataclass(frozen=True)
class InlineApplyOpArgs(RewritePattern):
    """
    Inlines apply op args into the callbacks.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        arg_mapping = zip(
            op.done_exchange.block.args[2:],
            op.args[-(len(op.done_exchange.block.args) - 2) :],
        )
        for block_arg, arg in [
            (op.done_exchange.block.args[0], op.field),
            *arg_mapping,
        ]:
            if isinstance(arg, OpResult) and arg.op.parent == op.parent:
                rewriter.insert_op(
                    new_arg := arg.op.clone(),
                    InsertPoint.at_start(op.done_exchange.block),
                )
                block_arg.replace_by(SSAValue.get(new_arg))


@dataclass(frozen=True)
class LowerCslStencil(ModulePass):
    """
    Lowers csl_stencil ops to csl and api calls.

    * `csl_stencil.access` are lowered to api call (emitting dsd) + UnrealizedConversionCastOp (converting dsd to
      memref).
    * The UnrealizedConversionCastOps are erased in the `memref-to-dsd` pass
    * `csl_stencil.apply` is lowered to an api call. Its two regions are placed into csl.funcs that are passed as
      callbacks to the api call.
    * `csl_stencil.yield` ops are lowered to `csl.return` as they terminate what are now callback functions with no
      return values.
    """

    name = "lower-csl-stencil"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerYieldOp(),
                    InlineApplyOpArgs(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAccessOp(),
                    LowerApplyOp(),
                ]
            )
        )
        module_pass.rewrite_module(op)
