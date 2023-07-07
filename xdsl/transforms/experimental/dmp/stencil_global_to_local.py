from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable, TypeVar, cast

from xdsl.dialects import arith, builtin, func, memref, mpi, scf, stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import Block, MLContext, Operation, OpResult, Region, SSAValue
from xdsl.irdl import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter
from xdsl.transforms.experimental.dmp.decompositions import (
    DomainDecompositionStrategy,
    GridSlice2d,
    HorizontalSlices2D,
)
from xdsl.transforms.experimental.stencil_shape_inference import (
    StencilShapeInferencePass,
)
from xdsl.utils.hints import isa

_T = TypeVar("_T", bound=Attribute)

_rank_dtype = builtin.i32


@dataclass
class ChangeStoreOpSizes(RewritePattern):
    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        assert all(
            integer_attr.data == 0 for integer_attr in op.lb.array.data
        ), "lb must be 0"
        shape: tuple[int, ...] = tuple(
            integer_attr.data for integer_attr in op.ub.array.data
        )
        new_shape = self.strategy.calc_resize(shape)
        op.ub = stencil.IndexAttr.get(*new_shape)


@dataclass
class AddHaloExchangeOps(RewritePattern):
    """
    This rewrite adds a `stencil.halo_exchange` after each `stencil.load` op
    """

    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.LoadOp, rewriter: PatternRewriter, /):
        swap_op = dmp.HaloSwapOp.get(op.res)
        swap_op.nodes = self.strategy.comm_layout()
        rewriter.insert_op_after_matched_op(swap_op)


@dataclass
class LowerHaloExchangeToMpi(RewritePattern):
    init: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.HaloSwapOp, rewriter: PatternRewriter, /):
        assert op.swaps is not None
        assert op.nodes is not None
        exchanges = list(op.swaps)

        assert isa(op.input_stencil.type, memref.MemRefType[Attribute])

        rewriter.replace_matched_op(
            list(
                generate_mpi_calls_for(
                    op.input_stencil,
                    exchanges,
                    op.input_stencil.type.element_type,
                    op.nodes,
                    emit_init=self.init,
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
            [true := arith.Constant.from_int_and_width(1, 1)],
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
            offset_v := arith.Constant.from_int_and_width(offset_in_axis, _rank_dtype),
            dest := arith.Addi(pos_in_axis, offset_v),
            # get the bound we need to check:
            bound := arith.Constant.from_int_and_width(
                0 if is_decrement else axis_size, _rank_dtype
            ),
            # comparison == true <=> we have a valid dest positon
            cond_val := arith.Cmpi.get(dest, bound, comparison),
        ],
        dest.result,
        cond_val.result,
    )


def _grid_coords_from_rank(
    my_rank: SSAValue, grid: dmp.NodeGrid
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Takes a rank and a dmp.grid, and returns operations to calculate
    the grid coordinates of the rank.
    """
    # a collection of all ops we want to return
    ret_ops: list[Operation] = []
    # the nodes coordinates in grid-space
    node_pos_nd: list[SSAValue] = []

    # first we translate the mpi rank into grid coordinates
    # (reversing the row major mapping)
    divide_by = 1
    for size in grid.as_tuple():
        ret_ops.extend(_div_mod(my_rank, divide_by, size))
        divide_by *= size
        node_pos_nd.append(ret_ops[-1].results[0])
    return ret_ops, node_pos_nd


def _generate_dest_rank_computation(
    my_rank: SSAValue,
    offsets: tuple[int, ...],
    grid: dmp.NodeGrid,
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
        cmp = arith.AndI(accumulated_cond_val, val)
        ret_ops.append(cmp)
        accumulated_cond_val = cmp.result

    # calculate rank of destination node from grid coords

    multiply_by = grid.as_tuple()[0]
    carry: SSAValue = dest_pos_nd[0]

    # dest rank: x * 1 + y * size[x] + z * size[x] * size[y] ...
    for pos, size in zip(dest_pos_nd[1:], grid.as_tuple()[1:]):
        fac = arith.Constant.from_int_and_width(multiply_by, _rank_dtype)
        val = arith.Muli(pos, fac)
        new_carry = arith.Addi(carry, val)
        carry = new_carry.result
        multiply_by *= size
        ret_ops.extend([fac, val, new_carry])

    return ret_ops, carry, accumulated_cond_val


def _div_mod(i: SSAValue, div: int, mod: int) -> list[Operation]:
    """
    Given (i, div, mod), generate ops that calculate (i / div) % mod

    The last returned operation has the final value as it's single result.
    """
    # these asserts should never trigger, but I'd rather have the compiler yell at me
    # when something goes wrong than see a spectacular runtime crash :D
    assert div > 0, "cannot work with negatives here!"
    assert mod > 0, "cannot work with negatives here!"
    # make sure we operate on an integer
    assert isinstance(i.type, builtin.IntegerType | builtin.IndexType)
    # we can use unsigned arithmetic here, because all we do is divide by positive
    # numbers and modulo positive numbers
    return [
        div_v := arith.Constant.from_int_and_width(div, i.type),
        div_res := arith.DivUI(i, div_v),
        mod_v := arith.Constant.from_int_and_width(mod, i.type),
        arith.RemUI(div_res, mod_v),
    ]


def generate_mpi_calls_for(
    source: SSAValue,
    exchanges: list[dmp.HaloExchangeDecl],
    dtype: Attribute,
    grid: dmp.NodeGrid,
    emit_init: bool = True,
) -> Iterable[Operation]:
    # call mpi init (this will be hoisted to function level)
    if emit_init:
        yield mpi.Init()
    # allocate request array
    # we need two request objects per exchange
    # one for the send, one for the recv
    req_cnt = arith.Constant.from_int_and_width(len(exchanges) * 2, builtin.i32)
    reqs = mpi.AllocateTypeOp.get(mpi.RequestType, req_cnt)
    # get comm rank
    rank = mpi.CommRank.get()
    # define static tag of 0
    # TODO: what is tag?
    tag = arith.Constant.from_int_and_width(0, builtin.i32)

    yield from (req_cnt, reqs, rank, tag)

    recv_buffers: list[tuple[dmp.HaloExchangeDecl, memref.Alloc, SSAValue]] = []

    for i, ex in enumerate(exchanges):
        # generate a temp buffer to store the data in
        reduced_size = [i for i in ex.size if i != 1]
        alloc_outbound = memref.Alloc.get(dtype, 64, reduced_size)
        alloc_outbound.memref.name_hint = f"send_buff_ex{i}"
        alloc_inbound = memref.Alloc.get(dtype, 64, reduced_size)
        alloc_inbound.memref.name_hint = f"recv_buff_ex{i}"
        yield from (alloc_outbound, alloc_inbound)

        # calc dest rank and check if it's in-bounds
        ops, dest_rank, is_in_bounds = _generate_dest_rank_computation(
            rank.rank, ex.neighbor, grid
        )
        yield from ops

        recv_buffers.append((ex, alloc_inbound, is_in_bounds))

        # get two unique indices
        cst_i = arith.Constant.from_int_and_width(i, builtin.i32)
        cst_in = arith.Constant.from_int_and_width(i + len(exchanges), builtin.i32)
        yield from (cst_i, cst_in)
        # from these indices, get request objects
        req_send = mpi.VectorGetOp.get(reqs, cst_i)
        req_recv = mpi.VectorGetOp.get(reqs, cst_in)
        yield from (req_send, req_recv)

        def then() -> Iterable[Operation]:
            # copy source area to outbound buffer
            yield from generate_memcpy(source, ex.source_area(), alloc_outbound.memref)
            # get ptr, count, dtype
            unwrap_out = mpi.UnwrapMemrefOp.get(alloc_outbound)
            unwrap_out.ptr.name_hint = f"send_buff_ex{i}_ptr"
            yield unwrap_out

            # isend call
            yield mpi.Isend.get(
                unwrap_out.ptr,
                unwrap_out.len,
                unwrap_out.type,
                dest_rank,
                tag,
                req_send,
            )

            # get ptr for receive buffer
            unwrap_in = mpi.UnwrapMemrefOp.get(alloc_inbound)
            unwrap_in.ptr.name_hint = f"recv_buff_ex{i}_ptr"
            yield unwrap_in
            # Irecv call
            yield mpi.Irecv.get(
                unwrap_in.ptr,
                unwrap_in.len,
                unwrap_in.type,
                dest_rank,
                tag,
                req_recv,
            )
            yield scf.Yield.get()

        def else_() -> Iterable[Operation]:
            # set the request object to MPI_REQUEST_NULL s.t. they are ignored
            # in the waitall call
            yield mpi.NullRequestOp.get(req_send)
            yield mpi.NullRequestOp.get(req_recv)
            yield scf.Yield.get()

        yield scf.If.get(
            is_in_bounds,
            [],
            Region([Block(then())]),
            Region([Block(else_())]),
        )

    # wait for all calls to complete
    yield mpi.Waitall.get(reqs.result, req_cnt.result)

    # start shuffling data into the main memref again
    for ex, buffer, cond_val in recv_buffers:
        yield scf.If.get(
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
                        + [scf.Yield.get()]
                    )
                ]
            ),
            Region([Block([scf.Yield.get()])]),
        )


def generate_memcpy(
    field: SSAValue, ex: dmp.HaloExchangeDecl, buffer: SSAValue, receive: bool = False
) -> list[Operation]:
    """
    This function generates a memcpy routine to copy over the parts
    specified by the `field` from `field` into `buffer`.

    If receive=True, it instead copy from `buffer` into the parts of
    `field` as specified by `ex`

    """
    assert isa(field.type, memref.MemRefType[Attribute])

    subview = memref.Subview.from_static_parameters(
        field, field.type, ex.offset, ex.size, [1] * len(ex.offset), reduce_rank=True
    )
    if receive:
        copy = memref.CopyOp(buffer, subview)
    else:
        copy = memref.CopyOp(subview, buffer)

    return [
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
        op: memref.Alloc
        | mpi.CommRank
        | mpi.AllocateTypeOp
        | mpi.UnwrapMemrefOp
        | mpi.Init,
        rewriter: Rewriter,
        /,
    ):
        if op in self.seen_ops:
            return
        self.seen_ops.add(op)

        # memref unwraps can always be moved to their allocation
        if isinstance(op, mpi.UnwrapMemrefOp) and isinstance(
            op.ref.owner, memref.Alloc
        ):
            op.detach()
            rewriter.insert_op_after(op.ref.owner, op)
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
        if isinstance(op, mpi.Init):
            # ignore multiple inits
            if parent in self.has_init:
                rewriter.erase_op(op)
                return
            self.has_init.add(parent)
            # add a finalize() call to the end of the function
            block = parent.regions[0].blocks[-1]
            return_op = block.last_op
            assert return_op is not None
            rewriter.insert_op_before(return_op, mpi.Finalize())

        ops = list(collect_args_recursive(op))
        for found_op in ops:
            found_op.detach()
            rewriter.insert_op_before(base, found_op)

    def get_matcher(
        self,
        worklist: list[
            memref.Alloc
            | mpi.CommRank
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemrefOp
            | mpi.Init
        ],
    ) -> Callable[[Operation], None]:
        """
        Returns a match() function that adds methods to a worklist
        if they satisfy some criteria.
        """

        def match(op: Operation):
            if isinstance(
                op,
                (
                    memref.Alloc,
                    mpi.CommRank,
                    mpi.AllocateTypeOp,
                    mpi.UnwrapMemrefOp,
                    mpi.Init,
                ),
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
            memref.Alloc
            | mpi.CommRank
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemrefOp
            | mpi.Init
        ] = list()
        matcher = self.get_matcher(worklist)
        for o in op.walk():
            matcher(o)

        # rewrite ops
        rewriter = Rewriter()
        for matched_op in worklist:
            self.rewrite(matched_op, rewriter)


_LOOP_INVARIANT_OPS = (arith.Constant, arith.Addi, arith.Muli)


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


@dataclass
class DmpSwapShapeInference:
    """
    Not a rewrite pattern, as it's a bit more involved.

    This is applied after stencil shape inference has run. It will find the
    HaloSwapOps again, and use the results of the shape inference pass
    to attach the swap declarations.
    """

    strategy: DomainDecompositionStrategy
    rewriter: Rewriter = field(default_factory=Rewriter)

    def match_and_rewrite(self, op: dmp.HaloSwapOp):
        core_lb: stencil.IndexAttr | None = None
        core_ub: stencil.IndexAttr | None = None

        for use in op.input_stencil.uses:
            if not isinstance(use.operation, stencil.ApplyOp):
                continue
            assert use.operation.res
            res_typ = cast(stencil.TempType[Attribute], use.operation.res[0].type)
            assert isinstance(res_typ.bounds, stencil.StencilBoundsAttr)
            core_lb = res_typ.bounds.lb
            core_ub = res_typ.bounds.ub
            break

        # this shouldn't have changed since the op was created!
        temp = op.input_stencil.type
        assert isa(temp, stencil.TempType[Attribute])
        assert isinstance(temp.bounds, stencil.StencilBoundsAttr)
        buff_lb = temp.bounds.lb
        buff_ub = temp.bounds.ub

        # fun fact: pyright does not understand this:
        # assert None not in (core_lb, core_ub, buff_lb, buff_ub)
        assert core_lb is not None
        assert core_ub is not None
        assert buff_lb is not None
        assert buff_ub is not None

        op.swaps = builtin.ArrayAttr(
            self.strategy.halo_exchange_defs(
                dmp.HaloShapeInformation.from_index_attrs(
                    buff_lb=buff_lb,
                    core_lb=core_lb,
                    buff_ub=buff_ub,
                    core_ub=core_ub,
                )
            )
        )

    def apply(self, module: builtin.ModuleOp):
        for op in module.walk():
            if isinstance(op, dmp.HaloSwapOp):
                self.match_and_rewrite(op)


@dataclass
class DmpDecompositionPass(ModulePass):
    """
    Represents a pass that takes a strategy as input
    """

    STRATEGIES: ClassVar[dict[str, type[DomainDecompositionStrategy]]] = {
        "2d-horizontal": HorizontalSlices2D,
        "2d-grid": GridSlice2d,
    }

    name = "dmp-decompose-2d"

    strategy: str

    slices: list[int]
    """
    Number of slices to decompose the input into
    """

    def get_strategy(self) -> DomainDecompositionStrategy:
        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self.STRATEGIES[self.strategy](self.slices)


@dataclass
class GlobalStencilToLocalStencil2DHorizontal(DmpDecompositionPass):
    """
    Decompose a stencil to apply to a local domain.

    This pass *replaces* stencil shape inference in a
    pass pipeline!
    """

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        strategy = self.get_strategy()

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ChangeStoreOpSizes(strategy),
                    AddHaloExchangeOps(strategy),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        # run the shape inference pass
        StencilShapeInferencePass().apply(ctx, op)

        DmpSwapShapeInference(strategy).apply(op)


@dataclass
class LowerHaloToMPI(ModulePass):
    name = "dmp-to-mpi"

    mpi_init: bool = True

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerHaloExchangeToMpi(
                        self.mpi_init,
                    ),
                ]
            )
        ).rewrite_module(op)
        MpiLoopInvariantCodeMotion().rewrite_module(op)
