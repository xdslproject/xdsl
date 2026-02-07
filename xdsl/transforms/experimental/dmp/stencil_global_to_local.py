from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import prod
from typing import ClassVar, cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, mpi, printf, scf, stencil
from xdsl.dialects.builtin import ContainerType
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import StencilToMemRefType

_rank_dtype = builtin.i32


@dataclass
class ChangeStoreOpSizes(RewritePattern):
    strategy: dmp.DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        assert all(
            integer_attr.data == 0 for integer_attr in op.bounds.lb.array.data
        ), "lb must be 0"
        shape: tuple[int, ...] = tuple(
            integer_attr.data for integer_attr in op.bounds.ub.array.data
        )
        new_shape = self.strategy.calc_resize(shape)
        op.bounds = stencil.StencilBoundsAttr.new(
            [
                stencil.IndexAttr.get(*(len(new_shape) * [0])),
                stencil.IndexAttr.get(*new_shape),
            ]
        )


@dataclass
class AddHaloExchangeOps(RewritePattern):
    """
    This rewrite adds a `stencil.halo_exchange` after each `stencil.load` op
    """

    strategy: dmp.DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.LoadOp, rewriter: PatternRewriter, /):
        swap_op = dmp.SwapOp.get(op.res, self.strategy)
        assert swap_op.swapped_values
        rewriter.insert_op(swap_op, InsertPoint.after(op))
        for use in tuple(op.res.uses):
            if use.operation is swap_op:
                continue
            use.operation.operands[use.index] = swap_op.swapped_values
            rewriter.handle_operation_modification(use.operation)


@dataclass
class LowerHaloExchangeToMpi(RewritePattern):
    init: bool
    debug_prints: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        exchanges = list(op.swaps)

        input_type = cast(ContainerType, op.input_stencil.type)

        rewriter.replace_op(
            op,
            list(
                generate_mpi_calls_for(
                    op.input_stencil,
                    exchanges,
                    input_type.get_element_type(),
                    op.strategy.comm_layout(),
                    emit_init=self.init,
                    emit_debug=self.debug_prints,
                )
            ),
            [],
        )


def _generate_single_axis_calc_and_check(
    pos_in_axis: SSAValue,
    offset_in_axis: int,
    axis_size: int,
) -> tuple[list[Operation], SSAValue, SSAValue]:
    """
    Given a position (in SSA), aand compile time known offset/size, generate the
    following operations:

    dest = pos_in_axis + offset_in_axis
    is_valid_axsis = (0 <= dest && dest < axis_size)

    Since we know the offset at compile time, we can skip one of the calculations.

    Returns the ops, and the ssa values that contain dest and is_valid_axis
    Return is ([ops], dest, is_valid)
    """
    # if the offset is zero, we can skip the comparison and just return true
    # because my_pos + 0 is always inbounds!
    if offset_in_axis == 0:
        return (
            [true := arith.ConstantOp.from_int_and_width(1, 1)],
            pos_in_axis,
            true.result,
        )

    # check if we are decrementing or increment the position here
    is_decrement = offset_in_axis < 0
    # very important that we use signed arithmetic here!
    # find the correct comparison for
    # is_valid_axsis = (0 <= dest && dest < axis_size)
    # since we know if we will be incrementing or decrementing, we can skip one of the
    # checks
    comparison = "sge" if is_decrement else "slt"

    return (
        [
            offset_v := arith.ConstantOp.from_int_and_width(
                offset_in_axis, _rank_dtype
            ),
            dest := arith.AddiOp(pos_in_axis, offset_v),
            # get the bound we need to check:
            bound := arith.ConstantOp.from_int_and_width(
                0 if is_decrement else axis_size, _rank_dtype
            ),
            # comparison == true <=> we have a valid dest positon
            cond_val := arith.CmpiOp(dest, bound, comparison),
        ],
        dest.result,
        cond_val.result,
    )


def _grid_coords_from_rank(
    my_rank: SSAValue, grid: dmp.RankTopoAttr
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Takes a rank and a dmp.grid, and returns operations to calculate
    the grid coordinates of the rank.
    """
    # a collection of all ops we want to return
    ret_ops: list[Operation] = []
    # the nodes coordinates in grid-space
    node_pos_nd: list[SSAValue] = []

    shape = grid.as_tuple()
    divisors = [prod(shape[i + 1 :]) for i in range(len(shape))]

    carry = my_rank
    for div in divisors:
        imm = arith.ConstantOp.from_int_and_width(div, builtin.i32)
        coord_i = arith.DivUIOp(carry, imm)
        carry = arith.RemUIOp(carry, imm)

        ret_ops.extend([imm, coord_i, carry])
        node_pos_nd.append(coord_i.result)

    return ret_ops, node_pos_nd


def _generate_dest_rank_computation(
    my_rank: SSAValue,
    offsets: tuple[int, ...],
    grid: dmp.RankTopoAttr,
) -> tuple[list[Operation], SSAValue, SSAValue]:
    """
    Takes the current rank, a tuple of offsets in grid coords, and a dmp.grid

    Calculates the dest rank, and comparisons if communication is in-bounds

    Returns a list of ops, the dest rank, and if it's in-bounds.

    Returns ([ops], dest_rank, is_in_bounds)
    """
    # calc grid coordinates
    ret_ops, node_pos_nd = _grid_coords_from_rank(my_rank, grid)

    # then we calculate the new coordinates:
    # save the condition vals somewhere
    condition_vals: list[SSAValue] = []
    # save the grid-coordinates of the destination rank
    dest_pos_nd: list[SSAValue] = []
    for pos_in_axis, offset_in_axis, axis_size in zip(
        node_pos_nd, offsets, grid.as_tuple()
    ):
        ops, dest, is_valid = _generate_single_axis_calc_and_check(
            pos_in_axis, offset_in_axis, axis_size
        )
        ret_ops.extend(ops)
        dest_pos_nd.append(dest)
        condition_vals.append(is_valid)

    # "and" all the condition vals
    accumulated_cond_val: SSAValue = condition_vals[0]
    for val in condition_vals[1:]:
        cmp = arith.AndIOp(accumulated_cond_val, val)
        ret_ops.append(cmp)
        accumulated_cond_val = cmp.result

    # calculate rank of destination node from grid coords

    carry: SSAValue = dest_pos_nd[-1]

    shape = grid.as_tuple()
    multiples = [prod(shape[i + 1 :]) for i in range(len(shape))]

    for pos, mul in zip(dest_pos_nd[:-1], multiples[:-1]):
        val = arith.ConstantOp.from_int_and_width(mul, builtin.i32)
        intermediate = arith.MuliOp(val, pos)
        carry_op = arith.AddiOp(carry, intermediate)
        carry = carry_op.result
        ret_ops.extend([val, intermediate, carry_op])

    return ret_ops, carry, accumulated_cond_val


def generate_mpi_calls_for(
    source: SSAValue,
    exchanges: list[dmp.ExchangeDeclarationAttr],
    dtype: Attribute,
    grid: dmp.RankTopoAttr,
    emit_init: bool = True,
    emit_debug: bool = False,
) -> Iterable[Operation]:
    # call mpi init (this will be hoisted to function level)
    if emit_init:
        yield mpi.InitOp()
    # allocate request array
    # we need two request objects per exchange
    # one for the send, one for the recv
    req_cnt = arith.ConstantOp.from_int_and_width(len(exchanges) * 2, builtin.i32)
    reqs = mpi.AllocateTypeOp(mpi.RequestType, req_cnt)
    # get comm rank
    rank = mpi.CommRankOp()
    # define static tag of 0
    tag = arith.ConstantOp.from_int_and_width(0, builtin.i32)

    yield from (req_cnt, reqs, rank, tag)

    recv_buffers: list[
        tuple[dmp.ExchangeDeclarationAttr, memref.AllocOp, SSAValue]
    ] = []

    for i, ex in enumerate(exchanges):
        # generate a temp buffer to store the data in
        reduced_size = [i for i in ex.size if i != 1]
        alloc_outbound = memref.AllocOp.get(dtype, 64, reduced_size)
        alloc_outbound.memref.name_hint = f"send_buff_ex{i}"
        alloc_inbound = memref.AllocOp.get(dtype, 64, reduced_size)
        alloc_inbound.memref.name_hint = f"recv_buff_ex{i}"
        yield from (alloc_outbound, alloc_inbound)

        # calc dest rank and check if it's in-bounds
        ops, dest_rank, is_in_bounds = _generate_dest_rank_computation(
            rank.rank, ex.neighbor, grid
        )
        yield from ops

        recv_buffers.append((ex, alloc_inbound, is_in_bounds))

        # get two unique indices
        cst_i = arith.ConstantOp.from_int_and_width(i, builtin.i32)
        cst_in = arith.ConstantOp.from_int_and_width(i + len(exchanges), builtin.i32)
        yield from (cst_i, cst_in)
        # from these indices, get request objects
        req_send = mpi.VectorGetOp(reqs, cst_i)
        req_recv = mpi.VectorGetOp(reqs, cst_in)
        yield from (req_send, req_recv)

        def then() -> Iterable[Operation]:
            # copy source area to outbound buffer
            yield from generate_memcpy(source, ex.source_area(), alloc_outbound.memref)
            # get ptr, count, dtype
            unwrap_out = mpi.UnwrapMemRefOp(alloc_outbound)
            unwrap_out.ptr.name_hint = f"send_buff_ex{i}_ptr"
            yield unwrap_out

            if emit_debug:
                yield printf.PrintFormatOp(
                    f"Rank {{}}: sending {ex.source_area()} -> {{}}\n", rank, dest_rank
                )

            # isend call
            yield mpi.IsendOp(
                unwrap_out.ptr,
                unwrap_out.len,
                unwrap_out.type,
                dest_rank,
                tag,
                req_send,
            )

            # get ptr for receive buffer
            unwrap_in = mpi.UnwrapMemRefOp(alloc_inbound)
            unwrap_in.ptr.name_hint = f"recv_buff_ex{i}_ptr"
            yield unwrap_in
            # Irecv call
            yield mpi.IrecvOp(
                unwrap_in.ptr,
                unwrap_in.len,
                unwrap_in.type,
                dest_rank,
                tag,
                req_recv,
            )
            yield scf.YieldOp()

        def else_() -> Iterable[Operation]:
            # set the request object to MPI_REQUEST_NULL s.t. they are ignored
            # in the waitall call
            yield mpi.NullRequestOp(req_send)
            yield mpi.NullRequestOp(req_recv)
            yield scf.YieldOp()

        yield scf.IfOp(
            is_in_bounds,
            [],
            Region([Block(then())]),
            Region([Block(else_())]),
        )

    # wait for all calls to complete
    yield mpi.WaitallOp(reqs.result, req_cnt.result)

    # start shuffling data into the main memref again
    for ex, buffer, cond_val in recv_buffers:
        yield scf.IfOp(
            cond_val,
            [],
            Region(
                [
                    Block(
                        list(
                            generate_memcpy(
                                source,
                                ex,
                                buffer.memref,
                                receive=True,
                            )
                        )
                        + [
                            printf.PrintFormatOp(
                                f"Rank {{}} receiving from {ex.neighbor}\n",
                                rank,
                            )
                        ]
                        * (1 if emit_debug else 0)
                        + [scf.YieldOp()]
                    )
                ]
            ),
            Region([Block([scf.YieldOp()])]),
        )


def generate_memcpy(
    field: SSAValue,
    ex: dmp.ExchangeDeclarationAttr,
    buffer: SSAValue,
    receive: bool = False,
) -> list[Operation]:
    """
    This function generates a memcpy routine to copy over the parts
    specified by the `field` from `field` into `buffer`.

    If receive=True, it instead copy from `buffer` into the parts of
    `field` as specified by `ex`

    """
    field_type = cast(stencil.FieldType[Attribute], field.type)
    assert isinstance(field_type.bounds, stencil.StencilBoundsAttr)
    memref_type = StencilToMemRefType(field_type)

    uc = builtin.UnrealizedConversionCastOp.get([field], result_type=[memref_type])

    memref_val = uc.results[0]

    offset = stencil.IndexAttr.get(*ex.offset) - field_type.bounds.lb

    subview = memref.SubviewOp.from_static_parameters(
        memref_val,
        memref_type,
        tuple(offset),
        ex.size,
        [1] * len(ex.offset),
        reduce_rank=True,
    )
    if receive:
        copy = memref.CopyOp(buffer, subview)
    else:
        copy = memref.CopyOp(subview, buffer)

    return [
        uc,
        subview,
        copy,
    ]


class MpiLoopInvariantCodeMotion:
    """
    THIS IS NOT A REWRITE PATTERN!

    This is a two-stage rewrite that modifies operations in a manner
    that is incompatible with the PatternRewriter!

    It implements a custom rewrite_module() method directly
    on the class.

    This rewrite moves all memref.allo, mpi.comm.rank, mpi.allocate
    and mpi.unwrap_memref ops and moves them "up" until it hits a
    func.func, and then places them *before* the op they appear in.
    """

    seen_ops: set[Operation]
    has_init: set[func.FuncOp]

    def __init__(self):
        self.seen_ops = set()
        self.has_init = set()

    def rewrite(
        self,
        op: (
            memref.AllocOp
            | mpi.CommRankOp
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemRefOp
            | mpi.InitOp
        ),
        rewriter: Rewriter,
        /,
    ):
        if op in self.seen_ops:
            return
        self.seen_ops.add(op)

        # memref unwraps can always be moved to their allocation
        if isinstance(op, mpi.UnwrapMemRefOp) and isinstance(
            op.ref.owner, memref.AllocOp
        ):
            op.detach()
            rewriter.insert_op(op, InsertPoint.after(op.ref.owner))
            return

        base = op
        parent = op.parent_op()
        # walk upwards until we hit a function
        while parent is not None and not isinstance(parent, func.FuncOp):
            base = parent
            parent = base.parent_op()

        # check that we did not run into "nowhere"
        assert parent is not None, "Expected MPI to be inside a func.FuncOp!"
        assert isinstance(parent, func.FuncOp)  # this must be true now

        # check that we "ascended"
        if base == op:
            return

        if not can_loop_invariant_code_move(op):
            return

        # if we move an mpi.init, generate a finalize()!
        if isinstance(op, mpi.InitOp):
            # ignore multiple inits
            if parent in self.has_init:
                rewriter.erase_op(op)
                return
            self.has_init.add(parent)
            # add a finalize() call to the end of the function
            block = parent.regions[0].blocks[-1]
            return_op = block.last_op
            assert return_op is not None
            rewriter.insert_op(mpi.FinalizeOp(), InsertPoint.before(return_op))

        ops = list(collect_args_recursive(op))
        for found_op in ops:
            found_op.detach()
            rewriter.insert_op(found_op, InsertPoint.before(base))

    def get_matcher(
        self,
        worklist: list[
            memref.AllocOp
            | mpi.CommRankOp
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemRefOp
            | mpi.InitOp
        ],
    ) -> Callable[[Operation], None]:
        """
        Returns a match() function that adds methods to a worklist
        if they satisfy some criteria.
        """

        def match(op: Operation):
            if isinstance(
                op,
                memref.AllocOp
                | mpi.CommRankOp
                | mpi.AllocateTypeOp
                | mpi.UnwrapMemRefOp
                | mpi.InitOp,
            ):
                worklist.append(op)

        return match

    def rewrite_module(self, op: builtin.ModuleOp):
        """
        Apply the rewrite to a module.

        We do a two-stage rewrite because we are modifying
        the operations we loop on them, which would throw of `op.walk`.
        """
        # collect all ops that should be rewritten
        worklist: list[
            memref.AllocOp
            | mpi.CommRankOp
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemRefOp
            | mpi.InitOp
        ] = list()
        matcher = self.get_matcher(worklist)
        for o in op.walk():
            matcher(o)

        # rewrite ops
        rewriter = Rewriter()
        for matched_op in worklist:
            self.rewrite(matched_op, rewriter)


_LOOP_INVARIANT_OPS = (arith.ConstantOp, arith.AddiOp, arith.MuliOp)


def can_loop_invariant_code_move(op: Operation):
    """
    This function walks the def-use chain up to see if all the args are
    "constant enough" to move outside the loop.

    This check is very conservative, but that means it definitely works!
    """

    for arg in op.operands:
        if not isinstance(arg, OpResult):
            print(f"{arg} is not opresult")
            return False
        if not isinstance(arg.owner, _LOOP_INVARIANT_OPS):
            print(f"{arg} is not loop invariant")
            return False
        if not can_loop_invariant_code_move(arg.owner):
            return False
    return True


def collect_args_recursive(op: Operation) -> Iterable[Operation]:
    """
    Collect the def-use chain "upwards" of an operation.
    Check with can_loop_invariant_code_move prior to using this!
    """
    for arg in op.operands:
        assert isinstance(arg, OpResult)
        yield from collect_args_recursive(arg.owner)
    yield op


@dataclass(frozen=True)
class DmpDecompositionPass(ModulePass, ABC):
    """
    Represents a pass that takes a strategy as input
    """


@dataclass(frozen=True)
class DistributeStencilPass(DmpDecompositionPass):
    """
    Decompose a stencil to apply to a local domain.

    This pass applies stencil shape inference!
    """

    name = "distribute-stencil"

    STRATEGIES: ClassVar[dict[str, type[dmp.GridSlice2dAttr | dmp.GridSlice3dAttr]]] = {
        "2d-grid": dmp.GridSlice2dAttr,
        "3d-grid": dmp.GridSlice3dAttr,
    }

    slices: tuple[int, ...]
    """
    Number of slices to decompose the input into
    """

    strategy: str
    """
    Name of the decomposition strategy to use, see STRATEGIES property for options
    """

    restrict_domain: bool = True
    """
    Apply the domain restriction (i.e. change the stencil.apply to operate on the
    local domain. If false, it assumes that the generated code is already local)
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        strategy = self.STRATEGIES[self.strategy](self.slices)

        rewrites: list[RewritePattern] = [
            AddHaloExchangeOps(strategy),
        ]

        if self.restrict_domain:
            rewrites.append(ChangeStoreOpSizes(strategy))

        PatternRewriteWalker(
            GreedyRewritePatternApplier(rewrites),
            apply_recursively=False,
        ).rewrite_module(op)


@dataclass(frozen=True)
class DmpToMpiPass(ModulePass):
    name = "dmp-to-mpi"

    mpi_init: bool = True

    generate_debug_prints: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerHaloExchangeToMpi(
                        self.mpi_init,
                        self.generate_debug_prints,
                    ),
                ]
            )
        ).rewrite_module(op)
        MpiLoopInvariantCodeMotion().rewrite_module(op)
