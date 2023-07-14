from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from xdsl.dialects.experimental import dmp


@dataclass
class DomainDecompositionStrategy(ABC):
    def __init__(self, _: list[int]):
        pass

    @abstractmethod
    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        raise NotImplementedError("SlicingStrategy must implement calc_resize!")

    @abstractmethod
    def halo_exchange_defs(
        self, dims: dmp.HaloShapeInformation
    ) -> Iterable[dmp.HaloExchangeDecl]:
        raise NotImplementedError("SlicingStrategy must implement halo_exchange_defs!")

    @abstractmethod
    def comm_layout(self) -> dmp.NodeGrid:
        raise NotImplementedError("SlicingStrategy must implement comm_count!")


@dataclass
class HorizontalSlices2D(DomainDecompositionStrategy):
    slices: int

    def __init__(self, slices: list[int]):
        super().__init__(slices)
        assert slices
        self.slices = slices[0]

    def __post_init__(self):
        assert self.slices > 1, "must slice into at least two pieces!"

    def comm_layout(self) -> dmp.NodeGrid:
        return dmp.NodeGrid([self.slices])

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        # slice on the y-axis
        assert len(shape) == 2, "HorizontalSlices2D only works on 2d fields!"
        assert (
            shape[1] % self.slices == 0
        ), "HorizontalSlices2D expects second dim to be divisible by number of slices!"

        return shape[0] // self.slices, shape[1]

    def halo_exchange_defs(
        self, dims: dmp.HaloShapeInformation
    ) -> Iterable[dmp.HaloExchangeDecl]:
        # upper halo exchange:
        yield dmp.HaloExchangeDecl(
            offset=(
                dims.buffer_start(dmp.DIM_X),
                dims.core_start(dmp.DIM_Y),
            ),
            size=(
                dims.halo_size(dmp.DIM_X),
                dims.core_size(dmp.DIM_Y),
            ),
            source_offset=(
                dims.halo_size(dmp.DIM_X),
                0,
            ),
            neighbor=[-1],
        )
        # lower halo exchange:
        yield dmp.HaloExchangeDecl(
            offset=(
                dims.core_end(dmp.DIM_X),
                dims.core_start(dmp.DIM_Y),
            ),
            size=(
                dims.halo_size(dmp.DIM_X),
                dims.core_size(dmp.DIM_Y),
            ),
            source_offset=(
                -dims.halo_size(dmp.DIM_X),
                0,
            ),
            neighbor=[1],
        )


@dataclass
class GridSlice2d(DomainDecompositionStrategy):
    """
    Takes a grid with two or more dimensions, slices it along the first two into equally
    sized segments.
    """

    topology: tuple[int, int]

    diagonals: bool = False

    def __post_init__(self):
        assert len(self.topology) >= 2, "GridSlice2d requires at least two dimensions"

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 2, "GridSlice2d requires at least two dimensions"
        for size, node_count in zip(shape, self.topology):
            assert (
                size % node_count == 0
            ), "GridSlice2d requires domain be neatly divisible by shape"
        return (
            *(size // node_count for size, node_count in zip(shape, self.topology)),
            *(size for size in shape[2:]),
        )

    def halo_exchange_defs(
        self, dims: dmp.HaloShapeInformation
    ) -> Iterable[dmp.HaloExchangeDecl]:
        # calculate values for the dimensions that were not decomposed
        residual_offsets = [0 for _ in range(2, dims.dims)]
        residual_sizes = [dims.buff_size(n) for n in range(2, dims.dims)]
        residual_source_offsets = [0 for _ in range(2, dims.dims)]

        # exchange to node "above" us on X axis direction
        yield dmp.HaloExchangeDecl(
            offset=(
                dims.buffer_start(dmp.DIM_X),
                dims.buffer_start(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                dims.buff_size(dmp.DIM_X),
                dims.halo_size(dmp.DIM_Y),
                *residual_sizes,
            ),
            source_offset=(
                0,
                dims.halo_size(dmp.DIM_Y),
                *residual_source_offsets,
            ),
            neighbor=(-1, 0),
        )
        # exchange to node "below" us on X axis direction
        yield dmp.HaloExchangeDecl(
            offset=(
                dims.buffer_start(dmp.DIM_X),
                dims.core_end(dmp.DIM_X),
                *residual_offsets,
            ),
            size=(
                dims.buff_size(dmp.DIM_X),
                dims.halo_size(dmp.DIM_Y, at_end=True),
                *residual_sizes,
            ),
            source_offset=(
                0,
                -dims.halo_size(dmp.DIM_Y, at_end=True),
                *residual_source_offsets,
            ),
            neighbor=(1, 0),
        )
        # exchange to node "left" of us on Y axis
        yield dmp.HaloExchangeDecl(
            offset=(
                dims.buffer_start(dmp.DIM_X),
                dims.buffer_start(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                dims.halo_size(dmp.DIM_X),
                dims.buff_size(dmp.DIM_Y),
                *residual_sizes,
            ),
            source_offset=(
                dims.halo_size(dmp.DIM_X),
                0,
                *residual_source_offsets,
            ),
            neighbor=(0, -1),
        )
        # exchange to node "right" of us on Y axis
        yield dmp.HaloExchangeDecl(
            offset=(
                dims.core_end(dmp.DIM_X),
                dims.buffer_start(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                dims.halo_size(dmp.DIM_X, at_end=True),
                dims.buff_size(dmp.DIM_Y),
                *residual_sizes,
            ),
            source_offset=(
                -dims.halo_size(dmp.DIM_X),
                0,
                *residual_source_offsets,
            ),
            neighbor=(0, 1),
        )
        # TOOD: add diagonals
        assert not self.diagonals

    def comm_layout(self) -> dmp.NodeGrid:
        return dmp.NodeGrid(self.topology)
