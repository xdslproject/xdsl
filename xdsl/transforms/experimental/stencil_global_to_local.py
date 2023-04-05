from dataclasses import dataclass
from typing import TypeVar, Iterable, ClassVar
from abc import ABC, abstractmethod

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import MLContext
from xdsl.irdl import Attribute
from xdsl.dialects import builtin
from xdsl.dialects.experimental import stencil

_T = TypeVar('_T', bound=Attribute)


@dataclass
class HaloExchangeDef:
    """
    This declares a region to be "halo-exchanged".
    The semantics define that the region specified by offset and size
    is the

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
    neighbor: tuple[int, ...]


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

    def __init__(self, buff_lb: tuple[int, ...], buff_ub: tuple[int, ...],
                 core_lb: tuple[int, ...], core_ub: tuple[int, ...]):
        assert len(buff_lb) == len(buff_ub) == len(core_lb) == len(
            core_ub), "Expected all args to be of the same dimensions!"
        self.dims = len(buff_lb)
        self.buff_lb = buff_lb
        self.buff_ub = buff_ub
        self.core_lb = core_lb
        self.core_ub = core_ub

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


@dataclass
class HorizontalSlices2D(SlicingStrategy):
    slices: int

    def __post_init__(self):
        assert self.slices > 1, "must slice into at least two pieces!"

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
            neighbor=(-1, 0),
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
            neighbor=(1, 0),
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
        swap_op = stencil.HaloSwapOp.get(op.field)
        rewriter.insert_op_before_matched_op(swap_op)


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
        pass
