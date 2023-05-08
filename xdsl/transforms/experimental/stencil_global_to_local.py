from dataclasses import dataclass
from typing import TypeVar, Iterable, ClassVar, Callable
from abc import ABC, abstractmethod
from math import prod

from xdsl.passes import ModulePass

from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter
from xdsl.ir import MLContext, Operation, SSAValue, Block, Region, OpResult
from xdsl.irdl import Attribute
from xdsl.dialects import builtin, mpi, memref, arith, scf, func
from xdsl.dialects.experimental import stencil

_T = TypeVar("_T", bound=Attribute)


@dataclass
class HaloExchangeDef:
    """
    This declares a region to be "halo-exchanged".
    The semantics define that the region specified by offset and size
    is the *received part*. To get the section that should be sent,
    use the source_area() method to get the source area.

     - offset gives the coordinates from the origin of the stencil field.
     - size gives the size of the buffer to be exchanged.
     - source_offset gives a translation (n-d offset) where the data should be
       read from that is exchanged with the other node.
     - neighbor gives the offset in rank to the node this data is to be
       exchanged with

    Example:

        offset = [4, 0]
        size   = [10, 1]
        source_offset = [0, 1]
        neighbor = -1

    To visualize:
    0   4         14
        xxxxxxxxxx    0
        oooooooooo    1

    Where `x` signifies the area that should be received,
    and `o` the area that should be read from.

    This data will be exchanged with the node of rank (my_rank -1)
    """

    offset: tuple[int, ...]
    size: tuple[int, ...]
    source_offset: tuple[int, ...]
    neighbor: int

    @property
    def elem_count(self) -> int:
        return prod(self.size)

    @property
    def dim(self) -> int:
        return len(self.offset)

    def source_area(self) -> "HaloExchangeDef":
        """
        Since a HaloExchangeDef by default specifies the area to receive into,
        this method returns the area that should be read from.
        """
        # we set source_offset to all zeor, so that repeated calls to source_area never return the dest area
        return HaloExchangeDef(
            offset=tuple(
                val + offs for val, offs in zip(self.offset, self.source_offset)
            ),
            size=self.size,
            source_offset=tuple(0 for _ in range(len(self.source_offset))),
            neighbor=self.neighbor,
        )


class DimsHelper:
    """
    Helper for getting various dimensions of an n-dimensional data array
    assuming we know outer dims, and halo.

    On the terminology used:

    In each dimension, we are given four points. we abbreviate them in
    annotations to an, bn, cn, dn, with n being the dimension. In 2d, these
    create the following pattern, higher dimensional examples can
    be derived from this:

    a0 b0          c0 d0
    +--+-----------+--+ a1
    |  |           |  |
    +--+-----------+--+ b1
    |  |           |  |
    |  |           |  |
    |  |           |  |
    |  |           |  |
    +--+-----------+--+ c1
    |  |           |  |
    +--+-----------+--+ d1

    We can now name these points:

         - a: buffer_start
         - b: core_start
         - c: core_end
         - d: buffer_end

    This class provides easy getters for these four.

    We can also define some common sizes on this object:

        - buff_size(n) = dn - an
        - core_size(n) = cn - bn
        - halo_size(n, start) = bn - an
        - halo_size(n, end  ) = dn - cn
    """

    dims: int
    buff_lb: tuple[int, ...]
    buff_ub: tuple[int, ...]
    core_lb: tuple[int, ...]
    core_ub: tuple[int, ...]

    DIM_X: ClassVar[int] = 0
    DIM_Y: ClassVar[int] = 1
    DIM_Z: ClassVar[int] = 2

    def __init__(self, op: stencil.HaloSwapOp):
        assert (
            op.buff_lb is not None
        ), "HaloSwapOp must be lowered after shape inference!"
        assert (
            op.buff_ub is not None
        ), "HaloSwapOp must be lowered after shape inference!"
        assert (
            op.core_lb is not None
        ), "HaloSwapOp must be lowered after shape inference!"
        assert (
            op.core_ub is not None
        ), "HaloSwapOp must be lowered after shape inference!"

        # translate everything to "memref" coordinates
        buff_lb = (op.buff_lb - op.buff_lb).as_tuple()
        buff_ub = (op.buff_ub - op.buff_lb).as_tuple()
        core_lb = (op.core_lb - op.buff_lb).as_tuple()
        core_ub = (op.core_ub - op.buff_lb).as_tuple()

        assert (
            len(buff_lb) == len(buff_ub) == len(core_lb) == len(core_ub)
        ), "Expected all args to be of the same length!"

        self.dims = len(buff_lb)
        self.buff_lb = buff_lb
        self.buff_ub = buff_ub
        self.core_lb = core_lb
        self.core_ub = core_ub

    # Helpers for specific positions:

    def buffer_start(self, dim: int):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.buff_lb[dim]

    def core_start(self, dim: int):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.core_lb[dim]

    def buffer_end(self, dim: int):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.core_ub[dim]

    def core_end(self, dim: int):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.buff_ub[dim]

    # Helpers for specific sizes:

    def buff_size(self, dim: int):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.buff_ub[dim] - self.buff_lb[dim]

    def core_size(self, dim: int):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.core_ub[dim] - self.core_lb[dim]

    def halo_size(self, dim: int, at_end: bool = False):
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        if at_end:
            return self.buff_ub[dim] - self.core_ub[dim]
        return self.core_lb[dim] - self.buff_lb[dim]


@dataclass
class DomainDecompositionStrategy(ABC):
    @abstractmethod
    def calc_resize(self, shape: tuple[int]) -> tuple[int]:
        raise NotImplementedError("SlicingStrategy must implement calc_resize!")

    @abstractmethod
    def halo_exchange_defs(self, dims: DimsHelper) -> Iterable[HaloExchangeDef]:
        raise NotImplementedError("SlicingStrategy must implement halo_exchange_defs!")

    @abstractmethod
    def comm_count(self) -> int:
        raise NotImplementedError("SlicingStrategy must implement comm_count!")


@dataclass
class HorizontalSlices2D(DomainDecompositionStrategy):
    slices: int

    def __post_init__(self):
        assert self.slices > 1, "must slice into at least two pieces!"

    def comm_count(self) -> int:
        return self.slices

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        # slice on the y-axis
        assert len(shape) == 2, "HorizontalSlices2D only works on 2d fields!"
        assert (
            shape[1] % self.slices == 0
        ), "HorizontalSlices2D expects second dim to be divisible by number of slices!"

        return shape[0], shape[1] // self.slices

    def halo_exchange_defs(self, dims: DimsHelper) -> Iterable[HaloExchangeDef]:
        # upper halo exchange:
        yield HaloExchangeDef(
            offset=(
                dims.core_start(dims.DIM_X),
                dims.buffer_start(dims.DIM_Y),
            ),
            size=(
                dims.core_size(dims.DIM_X),
                dims.halo_size(dims.DIM_Y),
            ),
            source_offset=(
                0,
                dims.halo_size(dims.DIM_Y),
            ),
            neighbor=-1,
        )
        # lower halo exchange:
        yield HaloExchangeDef(
            offset=(
                dims.core_start(dims.DIM_X),
                dims.core_end(dims.DIM_Y),
            ),
            size=(
                dims.core_size(dims.DIM_X),
                dims.halo_size(dims.DIM_Y),
            ),
            source_offset=(
                0,
                -dims.halo_size(dims.DIM_Y),
            ),
            neighbor=1,
        )


@dataclass
class ChangeStoreOpSizes(RewritePattern):
    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        assert all(
            integer_attr.value.data == 0 for integer_attr in op.lb.array.data
        ), "lb must be 0"
        shape: tuple[int, ...] = tuple(
            (integer_attr.value.data for integer_attr in op.ub.array.data)
        )
        new_shape = self.strategy.calc_resize(shape)
        op.ub = stencil.IndexAttr.get(*new_shape)


@dataclass
class AddHaloExchangeOps(RewritePattern):
    """
    This rewrite adds a `stencil.halo_exchange` before each `stencil.load` op
    """

    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.LoadOp, rewriter: PatternRewriter, /):
        swap_op = stencil.HaloSwapOp.get(op.res)
        rewriter.insert_op_after_matched_op(swap_op)


class GlobalStencilToLocalStencil2DHorizontal(ModulePass):
    name = "stencil-to-local-2d-horizontal"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        strategy = HorizontalSlices2D(2)

        gpra = GreedyRewritePatternApplier(
            [ChangeStoreOpSizes(strategy), AddHaloExchangeOps(strategy)]
        )

        PatternRewriteWalker(gpra, apply_recursively=False).rewrite_module(op)


@dataclass
class LowerHaloExchangeToMpi(RewritePattern):
    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.HaloSwapOp, rewriter: PatternRewriter, /):
        exchanges = list(self.strategy.halo_exchange_defs(DimsHelper(op)))
        assert isa(op.input_stencil.typ, memref.MemRefType[Attribute])
        rewriter.replace_matched_op(
            list(
                generate_mpi_calls_for(
                    op.input_stencil,
                    exchanges,
                    op.input_stencil.typ.element_type,
                    self.strategy,
                )
            ),
            [],
        )


def generate_mpi_calls_for(
    source: SSAValue,
    exchanges: list[HaloExchangeDef],
    dtype: Attribute,
    strat: DomainDecompositionStrategy,
) -> Iterable[Operation]:
    # call mpi init (this will be hoisted to function level)
    init = mpi.Init()
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

    yield from (init, req_cnt, reqs, rank, tag)

    recv_buffers: list[tuple[HaloExchangeDef, memref.Alloc, SSAValue]] = []

    for i, ex in enumerate(exchanges):
        neighbor_offset = arith.Constant.from_int_and_width(ex.neighbor, builtin.i32)
        neighbor_rank = arith.Addi(rank, neighbor_offset)
        yield from (neighbor_offset, neighbor_rank)

        # generate a temp buffer to store the data in
        alloc_outbound = memref.Alloc.get(dtype, 64, [ex.elem_count])
        alloc_inbound = memref.Alloc.get(dtype, 64, [ex.elem_count])
        yield from (alloc_outbound, alloc_inbound)

        # boundary condition:
        bound = arith.Constant.from_int_and_width(
            0 if ex.neighbor < 0 else strat.comm_count(), builtin.i32
        )
        comparison = "slt" if ex.neighbor < 0 else "sge"

        cond_val = arith.Cmpi.get(neighbor_rank, bound, comparison)
        yield from (bound, cond_val)

        recv_buffers.append((ex, alloc_inbound, cond_val.result))

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
            yield unwrap_out

            # isend call
            yield mpi.Isend.get(
                unwrap_out.ptr,
                unwrap_out.len,
                unwrap_out.typ,
                neighbor_rank,
                tag,
                req_send,
            )

            # get ptr for receive buffer
            unwrap_in = mpi.UnwrapMemrefOp.get(alloc_inbound)
            yield unwrap_in

            # Irecv call
            yield mpi.Irecv.get(
                unwrap_in.ptr,
                unwrap_in.len,
                unwrap_in.typ,
                neighbor_rank,
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
            cond_val,
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
                                ex.source_area(),
                                buffer.memref,
                                reverse=True,
                            )
                        )
                        + [scf.Yield.get()]
                    )
                ]
            ),
            Region([Block([scf.Yield.get()])]),
        )


def generate_memcpy(
    source: SSAValue, ex: HaloExchangeDef, dest: SSAValue, reverse: bool = False
) -> list[Operation]:
    """
    This function generates a memcpy routine to copy over the parts
    specified by the `ex` from `source` into `dest`.

    If reverse=True, it insteads copy from `dest` into the parts of
    `source` as specified by `ex`

    """
    assert ex.dim == 2, "Cannot handle non-2d case of memcpy yet!"
    x0 = arith.Constant.from_int_and_width(ex.offset[0], builtin.IndexType())
    x0.result.name_hint = "x0"
    y0 = arith.Constant.from_int_and_width(ex.offset[1], builtin.IndexType())
    y0.result.name_hint = "y0"
    x_len = arith.Constant.from_int_and_width(ex.size[0], builtin.IndexType())
    x_len.result.name_hint = "x_len"
    y_len = arith.Constant.from_int_and_width(ex.size[1], builtin.IndexType())
    y_len.result.name_hint = "y_len"
    cst0 = arith.Constant.from_int_and_width(0, builtin.IndexType())
    cst1 = arith.Constant.from_int_and_width(1, builtin.IndexType())

    # TODO: set to something like ex.size[1] < 8?
    unroll_inner = False

    # enable to get verbose information on what buffers are exchanged:
    # print("Generating{} memcpy from buff[{}:{},{}:{}]{}temp[{}:{}]".format(
    #    " unrolled" if unrolled else "",
    #    ex.offset[0], ex.offset[0] + ex.size[0],
    #    ex.offset[1], ex.offset[1] + ex.size[1],
    #    '<-' if reverse else '->',
    #    0, ex.elem_count
    # ))

    # only generate indices if we actually want to unroll
    if unroll_inner:
        indices = [
            arith.Constant.from_int_and_width(i, builtin.IndexType())
            for i in range(ex.offset[0], ex.offset[0] + ex.size[0])
        ]
    else:
        indices = []

    def loop_body_unrolled(i: SSAValue) -> Iterable[Operation]:
        """
        Generates last loop unrolled (not using scf.for)
        """
        dest_idx = arith.Muli.get(i, x_len)
        y = arith.Addi(i, y0)
        yield from (dest_idx, y)

        for x in indices:
            linearized_idx = arith.Addi(dest_idx, x)
            if reverse:
                load = memref.Load.get(dest, [linearized_idx])
                store = memref.Store.get(load, source, [x, y])
            else:
                load = memref.Load.get(source, [x, y])
                store = memref.Store.get(load, dest, [linearized_idx])
            yield from (linearized_idx, load, store)
        yield scf.Yield.get()

    def loop_body_with_for(i: SSAValue) -> Iterable[Operation]:
        """
        Generates last loop as scf.for
        """
        dest_idx = arith.Muli.get(i, x_len)
        y = arith.Addi(i, y0)
        yield from (dest_idx, y)

        def inner(j: SSAValue) -> Iterable[Operation]:
            x = arith.Addi(j, x0)
            linearized_idx = arith.Addi(dest_idx, j)
            if reverse:
                load = memref.Load.get(dest, [linearized_idx])
                store = memref.Store.get(load, source, [x, y])
            else:
                load = memref.Load.get(source, [x, y])
                store = memref.Store.get(load, dest, [linearized_idx])
            yield from (x, linearized_idx, load, store)
            # add an scf.yield at the end
            yield scf.Yield.get()

        yield scf.For.get(
            cst0,
            x_len,
            cst1,
            [],
            [Block.from_callable([builtin.IndexType()], inner)],  # type: ignore
        )

        yield scf.Yield.get()

    loop_body: Callable[[SSAValue], Iterable[Operation]] = (
        loop_body_unrolled if unroll_inner else loop_body_with_for
    )

    # TODO: make type annotations here aware that they can work with generators!
    loop = scf.For.get(
        cst0, y_len, cst1, [], Block.from_callable([builtin.IndexType()], loop_body)  # type: ignore
    )

    return [
        x0,
        y0,
        x_len,
        y_len,
        cst0,
        cst1,
        *indices,
        loop,
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
            print("{} is not opresult".format(arg))
            return False
        if not isinstance(arg.owner, _LOOP_INVARIANT_OPS):
            print("{} is not loop invariant".format(arg))
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
