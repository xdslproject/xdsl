from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

from xdsl.dialects.builtin import ArrayAttr, BoolAttr, IntAttr
from xdsl.dialects.experimental import dmp
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import ParameterDef, irdl_attr_definition


class DomainDecompositionStrategy(ParametrizedAttribute, ABC):
    def __init__(self, topo: tuple[int, ...]):
        super().__init__(
            [ArrayAttr(IntAttr(i) for i in topo), BoolAttr.from_int_and_width(0, 1)]
        )

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


@irdl_attr_definition
class GridSlice2d(DomainDecompositionStrategy):
    """
    Takes a grid with two or more dimensions, slices it along the first two into equally
    sized segments.
    """

    name = "dmp.grid_slice_2d"

    topology: ParameterDef[ArrayAttr[IntAttr]]

    diagonals: ParameterDef[BoolAttr]

    def _verify(self):
        assert len(self.topology) >= 2, "GridSlice2d requires at least two dimensions"

    def __init__(self, topo: tuple[int, ...]):
        super().new(
            [ArrayAttr(IntAttr(i) for i in topo), BoolAttr.from_int_and_width(0, 1)]
        )

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 2, "GridSlice2d requires at least two dimensions"
        for size, node_count in zip(shape, self.topology):
            assert (
                size % node_count.data == 0
            ), "GridSlice2d requires domain be neatly divisible by shape"
        return (
            *(
                size // node_count.data
                for size, node_count in zip(shape, self.topology)
            ),
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
        return dmp.RankTopoAttr(self.topology.data)


@irdl_attr_definition
class GridSlice3d(DomainDecompositionStrategy):
    """
    Takes a grid with two or more dimensions, slices it along the first three.
    """

    name = "dmp.grid_slice_2d"

    topology: ParameterDef[ArrayAttr[IntAttr]]

    diagonals: ParameterDef[BoolAttr]

    def _verify(self):
        assert len(self.topology) >= 3, "GridSlice3d requires at least three dimensions"

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 3, "GridSlice3d requires at least two dimensions"
        for size, node_count in zip(shape, self.topology):
            assert (
                size % node_count.data == 0
            ), "GridSlice3d requires domain be neatly divisible by shape"
        return (
            *(
                size // node_count.data
                for size, node_count in zip(shape, self.topology)
            ),
            *(size for size in shape[3:]),
        )

    def halo_exchange_defs(
        self, shape: dmp.ShapeAttr
    ) -> Iterable[dmp.ExchangeDeclarationAttr]:
        yield from _flat_face_exchanges_for_dim(shape, 0)

        yield from _flat_face_exchanges_for_dim(shape, 1)

        yield from _flat_face_exchanges_for_dim(shape, 2)

        # TOOD: add diagonals
        assert not self.diagonals.value.data

    def comm_layout(self) -> dmp.RankTopoAttr:
        return dmp.RankTopoAttr(self.topology.data)


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
