from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

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
        self, shape: dmp.HaloShapeInformation
    ) -> Iterable[dmp.HaloExchangeDecl]:
        raise NotImplementedError("SlicingStrategy must implement halo_exchange_defs!")

    @abstractmethod
    def comm_layout(self) -> dmp.NodeGrid:
        raise NotImplementedError("SlicingStrategy must implement comm_count!")


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
        self, shape: dmp.HaloShapeInformation
    ) -> Iterable[dmp.HaloExchangeDecl]:
        # calculate values for the dimensions that were not decomposed
        residual_offsets = [0 for _ in range(2, shape.dims)]
        residual_sizes = [shape.buff_size(n) for n in range(2, shape.dims)]
        residual_source_offsets = [0 for _ in range(2, shape.dims)]

        # exchange to node "above" us on X axis direction
        yield dmp.HaloExchangeDecl(
            offset=(
                shape.buffer_start(dmp.DIM_X),
                shape.buffer_start(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                shape.buff_size(dmp.DIM_X),
                shape.halo_size(dmp.DIM_Y),
                *residual_sizes,
            ),
            source_offset=(
                0,
                shape.halo_size(dmp.DIM_Y),
                *residual_source_offsets,
            ),
            neighbor=(-1, 0),
        )
        # exchange to node "below" us on X axis direction
        yield dmp.HaloExchangeDecl(
            offset=(
                shape.buffer_start(dmp.DIM_X),
                shape.core_end(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                shape.buff_size(dmp.DIM_X),
                shape.halo_size(dmp.DIM_Y, at_end=True),
                *residual_sizes,
            ),
            source_offset=(
                0,
                -shape.halo_size(dmp.DIM_Y, at_end=True),
                *residual_source_offsets,
            ),
            neighbor=(1, 0),
        )
        # exchange to node "left" of us on Y axis
        yield dmp.HaloExchangeDecl(
            offset=(
                shape.buffer_start(dmp.DIM_X),
                shape.buffer_start(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                shape.halo_size(dmp.DIM_X),
                shape.buff_size(dmp.DIM_Y),
                *residual_sizes,
            ),
            source_offset=(
                shape.halo_size(dmp.DIM_X),
                0,
                *residual_source_offsets,
            ),
            neighbor=(0, -1),
        )
        # exchange to node "right" of us on Y axis
        yield dmp.HaloExchangeDecl(
            offset=(
                shape.core_end(dmp.DIM_X),
                shape.buffer_start(dmp.DIM_Y),
                *residual_offsets,
            ),
            size=(
                shape.halo_size(dmp.DIM_X, at_end=True),
                shape.buff_size(dmp.DIM_Y),
                *residual_sizes,
            ),
            source_offset=(
                -shape.halo_size(dmp.DIM_X),
                0,
                *residual_source_offsets,
            ),
            neighbor=(0, 1),
        )
        # TOOD: add diagonals
        assert not self.diagonals

    def comm_layout(self) -> dmp.NodeGrid:
        return dmp.NodeGrid(self.topology)


@dataclass
class GridSlice3d(DomainDecompositionStrategy):
    """
    Takes a grid with two or more dimensions, slices it along the first three.
    """

    topology: tuple[int, int, int]

    diagonals: bool = False

    def __post_init__(self):
        assert len(self.topology) >= 3, "GridSlice3d requires at least three dimensions"

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 3, "GridSlice3d requires at least two dimensions"
        for size, node_count in zip(shape, self.topology):
            assert (
                size % node_count == 0
            ), "GridSlice3d requires domain be neatly divisible by shape"
        return (
            *(size // node_count for size, node_count in zip(shape, self.topology)),
            *(size for size in shape[3:]),
        )

    def halo_exchange_defs(
        self, shape: dmp.HaloShapeInformation
    ) -> Iterable[dmp.HaloExchangeDecl]:
        yield from _flat_face_exchanges_for_dim(shape, 0)

        yield from _flat_face_exchanges_for_dim(shape, 1)

        yield from _flat_face_exchanges_for_dim(shape, 2)

        # TOOD: add diagonals
        assert not self.diagonals

    def comm_layout(self) -> dmp.NodeGrid:
        return dmp.NodeGrid(self.topology)


def _flat_face_exchanges_for_dim(
    shape: dmp.HaloShapeInformation, axis: int
) -> tuple[dmp.HaloExchangeDecl, dmp.HaloExchangeDecl]:
    """
    Generate the two exchange delcarations to exchange the faces on the
    axis "axis".
    """
    dimensions = shape.dims
    assert axis <= dimensions

    def coords(where: Literal["start", "end"]):
        for d in range(dimensions):
            # for the dim we want to exchange, return either start or end halo region
            if d == axis:
                if where == "start":
                    # "start" halo goes from buffer start to core start
                    yield shape.buffer_start(d), shape.core_start(d)
                else:
                    # "end" halo goes from core end to buffer end
                    yield shape.core_end(d), shape.buffer_end(d)
            else:
                # for the sliced regions, "extrude" from core
                # this way we don't exchange edges
                yield shape.core_start(d), shape.core_end(d)

    ex1_coords = tuple(coords("end"))
    ex2_coords = tuple(coords("start"))

    return (
        # towards positive dim:
        dmp.HaloExchangeDecl.from_points(
            ex1_coords,
            axis,
            dir_sign=1,
        ),
        # towards negative dim:
        dmp.HaloExchangeDecl.from_points(
            ex2_coords,
            axis,
            dir_sign=-1,
        ),
    )
