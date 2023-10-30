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
        self, shape: dmp.ShapeAttr
    ) -> Iterable[dmp.ExchangeDeclarationAttr]:
        raise NotImplementedError("SlicingStrategy must implement halo_exchange_defs!")

    @abstractmethod
    def comm_layout(self) -> dmp.RankTopoAttr:
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
        self, shape: dmp.ShapeAttr
    ) -> Iterable[dmp.ExchangeDeclarationAttr]:
        yield from _flat_face_exchanges_for_dim(shape, 0)

        yield from _flat_face_exchanges_for_dim(shape, 1)

        # TOOD: add diagonals
        assert not self.diagonals

    def comm_layout(self) -> dmp.RankTopoAttr:
        return dmp.RankTopoAttr(self.topology)


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
        self, shape: dmp.ShapeAttr
    ) -> Iterable[dmp.ExchangeDeclarationAttr]:
        yield from _flat_face_exchanges_for_dim(shape, 0)

        yield from _flat_face_exchanges_for_dim(shape, 1)

        yield from _flat_face_exchanges_for_dim(shape, 2)

        # TOOD: add diagonals
        assert not self.diagonals

    def comm_layout(self) -> dmp.RankTopoAttr:
        return dmp.RankTopoAttr(self.topology)


def _flat_face_exchanges_for_dim(
    shape: dmp.ShapeAttr, axis: int
) -> tuple[dmp.ExchangeDeclarationAttr, dmp.ExchangeDeclarationAttr]:
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
        dmp.ExchangeDeclarationAttr.from_points(
            ex1_coords,
            axis,
            dir_sign=1,
        ),
        # towards negative dim:
        dmp.ExchangeDeclarationAttr.from_points(
            ex2_coords,
            axis,
            dir_sign=-1,
        ),
    )
