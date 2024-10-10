from dataclasses import dataclass
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import arith, func, memref
from xdsl.dialects.builtin import (
    ArrayAttr,
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
from xdsl.traits import is_side_effect_free
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr


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
                if not (
                    isinstance(arg.op, csl.LoadVarOp) or is_side_effect_free(arg.op)
                ):
                    raise ValueError(
                        "Can only promote csl.LoadVarOp or side_effect_free op"
                    )
                rewriter.insert_op(
                    new_arg := arg.op.clone(),
                    InsertPoint.at_start(op.done_exchange.block),
                )
                block_arg.replace_by(SSAValue.get(new_arg))


@dataclass(frozen=True)
class FullStencilAccessImmediateReductionOptimization(RewritePattern):
    """
    If an apply op accesses all points in the stencil shape *and* immediately performs a reduction,
    lower to an API call that iterates over all receive buffers at once. This requires setting up a
    4d dsd that disregards all but one dimension.

    The optimisation checks if it can be applied, and if so, sets up a new mem4d_dsd accumulator, lowers all
    relevant `csl_stencil.access` calls to a single mem4d_dsd API call, and replaces all relevant reduction ops
    with a single reduction op over the two mem4d_dsds.

    Note, if the optimisation is not applied, `csl_stencil.access` calls are left untouched to be handled by
    the `LowerAccessOp` pass instead and translated to individual mem1d_dsd API calls.

    The optimisation is applied on the `csl_stencil.apply.receive_chunk` region iff:
     * each point in the stencil shaped is accessed
     * each `csl_stencil.access` has exactly one use
     * each access is immediately processed by the same (type of) reduction op
     * each reduction op uses the same accumulator to store a result
     * each reduction op uses no inputs except from the above access ops
     * todo: the data of the accumulator is not itself an input of the reduction
     * todo: no other ops modify the accumulator in-between reduction ops
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        # check that apply is inside a csl_wrapper and retreive `pattern` (stencil arm length + self)
        if (wrapper := _get_module_wrapper(op)) is None:
            return
        pattern = wrapper.get_param_value("pattern").value.data

        # get csl_stencil.access ops and offsets
        access_ops: list[csl_stencil.AccessOp] = [
            a for a in op.receive_chunk.walk() if isinstance(a, csl_stencil.AccessOp)
        ]
        offsets = set(tuple(a.offset) for a in access_ops)

        # this rewrite only works if all points in the stencil shape are accessed
        if not self.is_full_2d_starshaped_access(offsets, pattern - 1):
            return

        # find potential 'reduction' ops
        reduction_ops: set[Operation] = set(
            u.operation for a in access_ops for u in a.result.uses
        )

        # check if reduction ops are of the same type
        red_op_ts = set(type(r) for r in reduction_ops)
        if len(red_op_ts) > 1 or (red_op_t := red_op_ts.pop()) not in [
            csl.FaddsOp,
            csl.FmulsOp,
        ]:
            return
        red_ops = cast(set[csl.BuiltinDsdOp], reduction_ops)

        # check: only apply rewrite if each access has exactly one use
        if any(len(a.result.uses) != 1 for a in access_ops):
            return

        # check: only apply rewrite if reduction ops use `access` ops only (plus one other, checked below)
        # note, we have already checked that each access op is only consumed once, which by implication is here
        red_args = set(arg for r in red_ops for arg in r.ops)
        nonaccess_args = red_args - set(a.result for a in access_ops)
        if len(nonaccess_args) > 1:
            return

        # check: only apply rewrite if the non-`access` op is an accumulator and the result param in all reduction ops
        accumulator = nonaccess_args.pop()
        if any(accumulator != r.ops[0] for r in red_ops):
            return

        if (
            not isattr(accumulator.type, memref.MemRefType)
            or not isinstance(op.accumulator, OpResult)
            or not isinstance(alloc := op.accumulator.op, memref.Alloc)
        ):
            raise ValueError("Pass needs to be run on memref types")

        # Set up new accumulator GetMemDsd, with 0-stride in `direction` and `distance` dimensions.
        # Effectively, this activates only the z-value dimension.
        dsd_t = csl.DsdType(csl.DsdKind.mem4d_dsd)
        direction_count = arith.Constant.from_int_and_width(4, 16)
        pattern = wrapper.get_program_param("pattern")
        chunk_size = wrapper.get_program_param("chunk_size")
        acc_dsd = csl.GetMemDsdOp.build(
            operands=[alloc, [direction_count, pattern, chunk_size]],
            result_types=[dsd_t],
            properties={"strides": ArrayAttr([IntegerAttr(i, 16) for i in [0, 0, 1]])},
        )
        new_acc = acc_dsd

        # If the accumulator is a subview at an offset, generate IncrementDsdOffset op (and index_cast).
        new_ops: list[Operation] = [direction_count, acc_dsd]
        if (
            isinstance(accumulator, OpResult)
            and isinstance(subview := accumulator.op, memref.Subview)
            and subview.source == op.receive_chunk.block.args[2]
        ):
            assert isa(subview.source.type, memref.MemRefType[Attribute])
            new_ops.append(
                cast_op := arith.IndexCastOp(subview.offsets[0], csl.i16_value)
            )
            new_ops.append(
                new_acc := csl.IncrementDsdOffsetOp.build(
                    operands=[acc_dsd, cast_op],
                    properties={"elem_type": subview.source.type.get_element_type()},
                    result_types=[dsd_t],
                )
            )

        # get dsd iterator over all points in stencil
        full_stencil_dsd = csl.MemberCallOp(
            "getRecvBufDsd", dsd_t, wrapper.get_program_import("stencil_comms.csl"), []
        )

        # rebuild compute func
        reduction_op = red_op_t.build(operands=[[new_acc, new_acc, full_stencil_dsd]])

        rewriter.insert_op(
            [*new_ops, full_stencil_dsd, reduction_op],
            InsertPoint.after(list(red_ops)[-1]),
        )

        for e in [*access_ops, *red_ops]:
            rewriter.erase_op(e, safe_erase=False)

    @staticmethod
    def is_full_2d_starshaped_access(
        offsets: set[tuple[int, ...]], max_offset: int
    ) -> bool:
        """Returns iff the offsets cover all points in a 2d star-shape without the (0,0) point."""
        x_set = set((x, 0) for x in range(-max_offset, max_offset + 1))
        y_set = set((0, y) for y in range(-max_offset, max_offset + 1))
        return offsets == x_set ^ y_set


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
                    FullStencilAccessImmediateReductionOptimization(),
                    LowerAccessOp(),
                    LowerApplyOp(),
                ]
            )
        )
        module_pass.rewrite_module(op)
