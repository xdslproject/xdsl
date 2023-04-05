from dataclasses import dataclass
from typing import TypeVar, Iterable, ClassVar
from abc import ABC, abstractmethod
from math import prod

from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import MLContext, Operation, SSAValue, Block, Region
from xdsl.irdl import Attribute
from xdsl.dialects import builtin, mpi, memref, arith, scf
from xdsl.dialects.experimental import stencil

_T = TypeVar('_T', bound=Attribute)


@dataclass
class HaloExchangeDef:
    """
    This declares a region to be "halo-exchanged".
    The semantics define that the region specified by offset and size
    is the *received part*. To get the section that should be sent,
    use the source_area() method to get the source area.

    offset gives the coordinates from the origin of the stencil field.

    size gives the size of the buffer to be exchanged.

    source_offset gives a translation (n-d offset) where the data should be
    read from that is exchanged with the other node.

    Finally, neighbor gives the n-dimensional offset to the node with whom
    this edge should be exchanged.

    Example:

        offset = [4, 0]
        size   = [10, 1]
        source_offset = [0, 1]
        neighbor = [-1, 0]

    To visualize:
    a0  b0        c0
        xxxxxxxxxx    a1
        oooooooooo    b1

    Where `x` signifies the area that should be received,
    and `o` the area that should be read from.

    This will be sent to the neighbor of 0 * k + (-1) where k is the number
    of nodes per row of data.
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

    def source_area(self) -> 'HaloExchangeDef':
        """
        Since a HaloExchangeDef by default specifies the area to receive into,
        this method returns the area that should be read from.
        """
        # we set source_offset to all zeor, so that repeated calls to source_area never return the dest area
        return HaloExchangeDef(
            offset=tuple(
                val - offs
                for val, offs in zip(self.offset, self.source_offset)),
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

    def __init__(
        self,
        buff_lb: tuple[int, ...],
        buff_ub: tuple[int, ...],
        core_lb: tuple[int, ...],
        core_ub: tuple[int, ...],
    ):
        assert len(buff_lb) == len(buff_ub) == len(core_lb) == len(core_ub), \
            "Expected all args to be of the same dimensions!"
        self.dims = len(buff_lb)
        self.buff_lb = buff_lb
        self.buff_ub = buff_ub
        self.core_lb = core_lb
        self.core_ub = core_ub

    @staticmethod
    def from_halo_swap_op(op: stencil.HaloSwapOp):
        return DimsHelper(
            op.buff_lb.as_tuple(),
            op.buff_ub.as_tuple(),
            op.core_lb.as_tuple(),
            op.core_ub.as_tuple(),
        )

    # positions

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

    # sizes

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
class SlicingStrategy(ABC):

    @abstractmethod
    def calc_resize(self, shape: tuple[int]) -> tuple[int]:
        raise NotImplementedError(
            "SlicingStrategy must implement calc_resize!")

    @abstractmethod
    def halo_exchange_defs(self,
                           dims: DimsHelper) -> Iterable[HaloExchangeDef]:
        raise NotImplementedError(
            "SlicingStrategy must implement halo_exchange_defs!")

    @abstractmethod
    def comm_count(self) -> int:
        raise NotImplementedError("SlicingStrategy must implement comm_count!")


@dataclass
class HorizontalSlices2D(SlicingStrategy):
    slices: int

    def __post_init__(self):
        assert self.slices > 1, "must slice into at least two pieces!"

    def comm_count(self) -> int:
        return self.slices

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        # slice on the last dimension only
        assert len(shape) == 2, \
            "HorizontalSlices2D only works on 2d fields!"
        assert shape[1] % self.slices == 0, \
            "HorizontalSlices2D expects second dim to be divisible by number of slices!"

        return shape[0], shape[1] // self.slices

    def halo_exchange_defs(self,
                           dims: DimsHelper) -> Iterable[HaloExchangeDef]:
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
    strategy: SlicingStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter,
                          /):
        assert all(integer_attr.value.data == 0
                   for integer_attr in op.lb.array.data), "lb must be 0"
        shape: tuple[int, ...] = tuple(
            (integer_attr.value.data for integer_attr in op.ub.array.data))
        new_shape = self.strategy.calc_resize(shape)
        op.ub = stencil.IndexAttr.get(*new_shape)


@dataclass
class AddHaloExchangeOps(RewritePattern):
    """
    This rewrite adds a `stencil.halo_exchange` before each `stencil.load` op
    """
    strategy: SlicingStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.LoadOp, rewriter: PatternRewriter,
                          /):
        swap_op = stencil.HaloSwapOp.get(op.res)
        rewriter.insert_op_after_matched_op(swap_op)


def global_stencil_to_local_stencil_2d_horizontal(ctx: MLContext | None,
                                                  module: builtin.ModuleOp):
    strategy = HorizontalSlices2D(2)

    gpra = GreedyRewritePatternApplier(
        [ChangeStoreOpSizes(strategy),
         AddHaloExchangeOps(strategy)])

    PatternRewriteWalker(gpra, apply_recursively=False).rewrite_module(module)


@dataclass
class LowerHaloExchangeToMpi(RewritePattern):
    strategy: SlicingStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.HaloSwapOp,
                          rewriter: PatternRewriter, /):
        exchanges = list(
            self.strategy.halo_exchange_defs(DimsHelper.from_halo_swap_op(op)))
        assert isa(op.input_stencil.typ, memref.MemRefType[Attribute])
        rewriter.replace_matched_op(
            list(
                generate_mpi_calls_for(
                    op.input_stencil,
                    exchanges,
                    op.input_stencil.typ.element_type,
                    self.strategy,
                )),
            [],
        )


def generate_mpi_calls_for(source: SSAValue, exchanges: list[HaloExchangeDef],
                           dtype: Attribute,
                           strat: SlicingStrategy) -> Iterable[Operation]:
    # allocate request array
    req_cnt = arith.Constant.from_int_and_width(
        len(exchanges) * 2, builtin.i32)
    reqs = mpi.AllocateTypeOp.get(mpi.RequestType, req_cnt)

    # get comm rank
    rank = mpi.CommRank.get()
    # define static tag of 0
    # TODO: what is tag?
    tag = arith.Constant.from_int_and_width(0, builtin.i32)

    yield from (req_cnt, reqs, rank, tag)

    recv_buffers: list[tuple[HaloExchangeDef, memref.Alloc, SSAValue]] = []

    for i, ex in enumerate(exchanges):
        neighbor_offset = arith.Constant.from_int_and_width(
            ex.neighbor, builtin.i32)
        neighbor_rank = arith.Addi.get(rank, neighbor_offset)
        yield from (neighbor_offset, neighbor_rank)

        # generate a temp buffer to store the data in
        alloc_outbound = memref.Alloc.get(dtype, 64, (ex.elem_count, ))
        alloc_inbound = memref.Alloc.get(dtype, 64, (ex.elem_count, ))
        yield from (alloc_outbound, alloc_inbound)

        # boundary condition:
        bound = arith.Constant.from_int_and_width(
            0 if ex.neighbor < 0 else strat.comm_count(), builtin.i32)
        comparison = 'slt' if ex.neighbor < 0 else 'sge'

        cond_val = arith.Cmpi.get(neighbor_rank, bound, comparison)
        yield from (bound, cond_val)

        recv_buffers.append((ex, alloc_inbound, cond_val.result))

        def body():
            # copy source area to outbound buffer
            yield from generate_memcpy(source, ex.source_area(),
                                       alloc_outbound.memref)
            # get ptr, count, dtype
            unwrap_out = mpi.UnwrapMemrefOp.get(alloc_outbound)
            yield unwrap_out
            # get two unique indices
            cst_i = arith.Constant.from_int_and_width(i, builtin.i32)
            cst_in = arith.Constant.from_int_and_width(i + len(exchanges),
                                                       builtin.i32)
            yield from (cst_i, cst_in)
            # from these indices, get request objects
            req_send = mpi.VectorGetOp.get(reqs, cst_i)
            req_recv = mpi.VectorGetOp.get(reqs, cst_in)
            yield from (req_send, req_recv)

            # isend call
            yield mpi.Isend.get(unwrap_out.ptr, unwrap_out.len, unwrap_out.typ,
                                neighbor_rank, tag, req_send)

            # get ptr for receive buffer
            unwrap_in = mpi.UnwrapMemrefOp.get(alloc_inbound)
            yield unwrap_in

            # Irecv call
            yield mpi.Irecv.get(unwrap_in.ptr, unwrap_in.len, unwrap_in.typ,
                                neighbor_rank, tag, req_send)
            yield scf.Yield.get()

        yield scf.If.get(
            cond_val,
            [],
            Region.from_operation_list(list(body())),
            Region.from_operation_list([scf.Yield.get()]),
        )

    # wait for all calls to complete
    yield mpi.Waitall.get(reqs, req_cnt)

    # start shuffling data into the main memref again
    for ex, buffer, cond_val in recv_buffers:
        yield scf.If.get(
            cond_val,
            [],
            Region.from_operation_list(
                list(
                    generate_memcpy(
                        source,
                        ex.source_area(),
                        buffer.memref,
                        reverse=True,
                    )) + [scf.Yield.get()]),
            Region.from_operation_list([scf.Yield.get()]),
        )


def generate_dest_rank_conditional(cond_val: SSAValue,
                                   body: Iterable[Operation]):
    return


def generate_memcpy(source: SSAValue,
                    ex: HaloExchangeDef,
                    dest: SSAValue,
                    reverse: bool = False) -> list[Operation]:
    """
    This function generates a memcpy routine to copy over the parts
    specified by the `ex` from `source` into `dest`.

    If reverse=True, it insteads copy from `dest` into the parts of
    `source` as specified by `ex`

    """
    assert ex.dim == 2, "Cannot handle non-2d case of memcpy yet!"
    y0 = arith.Constant.from_int_and_width(ex.offset[1], builtin.IndexType())
    x_len = arith.Constant.from_int_and_width(ex.size[0], builtin.IndexType())
    y_len = arith.Constant.from_int_and_width(ex.size[1], builtin.IndexType())
    cst0 = arith.Constant.from_int_and_width(0, builtin.IndexType())
    cst1 = arith.Constant.from_int_and_width(1, builtin.IndexType())

    indices = [
        arith.Constant.from_int_and_width(i, builtin.IndexType())
        for i in range(ex.offset[0], ex.offset[0] + ex.size[0])
    ]

    def loop_body(y: SSAValue):
        linearized_y = arith.Muli.get(y, x_len)
        y_with_offset = arith.Addi.get(y, y0)
        yield from (linearized_y, y_with_offset)

        for x in indices:
            linearized_idx = arith.Addi.get(linearized_y, x)
            if reverse:
                load = memref.Load.get(dest, [linearized_idx])
                store = memref.Store.get(load, source, [x, y_with_offset])
            else:
                load = memref.Load.get(source, [x, y_with_offset])
                store = memref.Store.get(load, dest, [linearized_idx])
            yield from (linearized_idx, load, store)
        yield scf.Yield()

    loop = scf.For.get(cst0, y_len, cst1, [],
                       Block.from_callable([builtin.IndexType()], loop_body))

    return [
        y0,
        x_len,
        y_len,
        cst0,
        cst1,
        *indices,
        loop,
    ]
