"""
A dialect for handling distributed memory parallelism (DMP).

This is xDSL only for now.

This dialect aims to provide the tools necessary to facilitate the creation
and lowering of stencil (and other) computations in a manner that
makes them run on node clusters.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Sequence
from math import prod
from typing import Literal, cast

from xdsl.dialects import builtin, stencil
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    EffectInstance,
    HasShapeInferencePatternsTrait,
    MemoryEffect,
    MemoryEffectKind,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

# helpers for named dimensions:
DIM_X = 0
DIM_Y = 1
DIM_Z = 2


@irdl_attr_definition
class ExchangeDeclarationAttr(ParametrizedAttribute):
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

    name = "dmp.exchange"

    offset_: builtin.DenseArrayBase[builtin.I64]
    size_: builtin.DenseArrayBase[builtin.I64]
    source_offset_: builtin.DenseArrayBase[builtin.I64]
    neighbor_: builtin.DenseArrayBase[builtin.I64]

    def __init__(
        self,
        offset: Sequence[int],
        size: Sequence[int],
        source_offset: Sequence[int],
        neighbor: Sequence[int],
    ):
        data_type = builtin.i64
        super().__init__(
            builtin.DenseArrayBase.from_list(data_type, offset),
            builtin.DenseArrayBase.from_list(data_type, size),
            builtin.DenseArrayBase.from_list(data_type, source_offset),
            builtin.DenseArrayBase.from_list(data_type, neighbor),
        )

    @classmethod
    def from_points(
        cls,
        points: Sequence[tuple[int, int]],
        dim: int,
        dir_sign: Literal[1, -1],
        neighbor_offset: int = 1,
    ):
        sizes = tuple(e - s for s, e in points)
        return cls(
            # get starting points
            tuple(s for s, _ in points),
            # calculated sizes
            sizes,
            # source_offset (opposite of exchange direction)
            tuple(
                0 if d != dim else -1 * dir_sign * sizes[dim] * neighbor_offset
                for d in range(len(sizes))
            ),
            # direction
            tuple(
                0 if d != dim else dir_sign * neighbor_offset for d in range(len(sizes))
            ),
        )

    @property
    def offset(self) -> tuple[int, ...]:
        return self.offset_.get_values()

    @property
    def size(self) -> tuple[int, ...]:
        return self.size_.get_values()

    @property
    def source_offset(self) -> tuple[int, ...]:
        return self.source_offset_.get_values()

    @property
    def neighbor(self) -> tuple[int, ...]:
        return self.neighbor_.get_values()

    @property
    def elem_count(self) -> int:
        return prod(self.size)

    @property
    def dims(self) -> int:
        """
        number of dimensions of the grid
        """
        return len(self.size)

    def source_area(self) -> ExchangeDeclarationAttr:
        """
        Since a HaloExchangeDef by default specifies the area to receive into,
        this method returns the area that should be read from.
        """
        # we set source_offset to all zero, so that repeated calls to source_area never
        # return the dest area
        return ExchangeDeclarationAttr(
            offset=tuple(
                val + offs for val, offs in zip(self.offset, self.source_offset)
            ),
            size=self.size,
            source_offset=tuple(0 for _ in range(len(self.source_offset))),
            neighbor=self.neighbor,
        )

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("at ")
            with printer.in_square_brackets():
                printer.print_list(self.offset, printer.print_int)
            printer.print_string(" size ")
            with printer.in_square_brackets():
                printer.print_list(self.size, printer.print_int)
            printer.print_string(" source offset ")
            with printer.in_square_brackets():
                printer.print_list(self.source_offset, printer.print_int)
            printer.print_string(" to ")
            with printer.in_square_brackets():
                printer.print_list(self.neighbor, printer.print_int)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        parser.parse_characters("at")
        offset = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        parser.parse_characters("size")
        size = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        parser.parse_characters("source")
        parser.parse_characters("offset")
        source_offset = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        parser.parse_characters("to")
        to = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        parser.parse_characters(">")

        return [
            builtin.DenseArrayBase.from_list(builtin.i64, x)
            for x in (offset, size, source_offset, to)
        ]


@irdl_attr_definition
class ShapeAttr(ParametrizedAttribute):
    """
    This represents shape information that is attached to halo operations.

    On the terminology used:

    In each dimension, we are given four points. We abbreviate them in
    annotations to an, bn, cn, dn, with n being the dimension. In 2d, these
    create the following pattern, higher dimensional examples can
    be derived from this:

    a1 b1          c1 d1
    +--+-----------+--+ a0
    |  |           |  |
    +--+-----------+--+ b0
    |  |           |  |
    |  |           |  |
    |  |           |  |
    |  |           |  |
    +--+-----------+--+ c0
    |  |           |  |
    +--+-----------+--+ d0

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

    name = "dmp.shape_with_halo"

    buff_lb_: builtin.DenseArrayBase[builtin.I64]
    buff_ub_: builtin.DenseArrayBase[builtin.I64]
    core_lb_: builtin.DenseArrayBase[builtin.I64]
    core_ub_: builtin.DenseArrayBase[builtin.I64]

    @property
    def buff_lb(self) -> tuple[int, ...]:
        data = self.buff_lb_.get_values()
        return data

    @property
    def buff_ub(self) -> tuple[int, ...]:
        data = self.buff_ub_.get_values()
        return data

    @property
    def core_lb(self) -> tuple[int, ...]:
        data = self.core_lb_.get_values()
        return data

    @property
    def core_ub(self) -> tuple[int, ...]:
        data = self.core_ub_.get_values()
        return data

    @property
    def dims(self) -> int:
        """
        Number of axis of the data (len(shape))
        """
        return len(self.core_ub)

    @staticmethod
    def from_index_attrs(
        buff_lb: stencil.IndexAttr | Sequence[int],
        core_lb: stencil.IndexAttr | Sequence[int],
        core_ub: stencil.IndexAttr | Sequence[int],
        buff_ub: stencil.IndexAttr | Sequence[int],
    ):
        data_type = builtin.i64
        return ShapeAttr(
            *(
                builtin.DenseArrayBase.from_list(data_type, tuple(data))
                for data in (buff_lb, buff_ub, core_lb, core_ub)
            )
        )

    def buffer_start(self, dim: int) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.buff_lb[dim]

    def core_start(self, dim: int) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.core_lb[dim]

    def buffer_end(self, dim: int) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.buff_ub[dim]

    def core_end(self, dim: int) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.core_ub[dim]

    # Helpers for specific sizes:

    def buff_size(self, dim: int) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.buff_ub[dim] - self.buff_lb[dim]

    def core_size(self, dim: int) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        return self.core_ub[dim] - self.core_lb[dim]

    def halo_size(self, dim: int, at_end: bool = False) -> int:
        assert dim < self.dims, f"The given DimsHelper only has {self.dims} dimensions"
        if at_end:
            return self.buff_ub[dim] - self.core_ub[dim]
        return self.core_lb[dim] - self.buff_lb[dim]

    # parsing / printing

    def print_parameters(self, printer: Printer) -> None:
        dims = zip(self.buff_lb, self.core_lb, self.core_ub, self.buff_ub)
        printer.print_string("<")
        printer.print_string("x".join(f"{list(vals)}" for vals in dims))
        printer.print_string(">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        """
        Parses the attribute, the format of it is:

        #dmp.shape_with_halo<[a0,b0,c0,d0]x[a1,b1,c1,d1]x...>

        so different from the way it's stored internally.

        This decision was made to improve readability.
        """
        parser.parse_characters("<")
        buff_lb: list[int] = []
        buff_ub: list[int] = []
        core_lb: list[int] = []
        core_ub: list[int] = []

        while True:
            parser.parse_characters("[")
            buff_lb.append(parser.parse_integer())
            parser.parse_characters(",")
            core_lb.append(parser.parse_integer())
            parser.parse_characters(",")
            core_ub.append(parser.parse_integer())
            parser.parse_characters(",")
            buff_ub.append(parser.parse_integer())
            parser.parse_characters("]")
            if parser.parse_optional_characters("x") is None:
                break
        parser.parse_characters(">")

        data_type = builtin.i64
        return [
            builtin.DenseArrayBase.from_list(data_type, data)
            for data in (buff_lb, buff_ub, core_lb, core_ub)
        ]


@irdl_attr_definition
class RankTopoAttr(ParametrizedAttribute):
    """
    This attribute specifies the node layout used to distribute the computation.

    dmp.grid<3x3> means nine ranks organized in a 3x3 grid.

    This allows for higher-dimensional grids as well, e.g. dmp.grid<3x3x3> for
    3-dimensional data.
    """

    name = "dmp.topo"

    shape: builtin.DenseArrayBase[builtin.I64]

    def __init__(self, shape: Sequence[int]):
        if len(shape) < 1:
            raise ValueError("dmp.grid must have at least one dimension!")
        super().__init__(builtin.DenseArrayBase.from_list(builtin.i64, shape))

    def as_tuple(self) -> tuple[int, ...]:
        shape = self.shape.get_values()
        return shape

    def node_count(self) -> int:
        return prod(self.as_tuple())

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        shape: list[int] = [
            parser.parse_integer(allow_negative=False, allow_boolean=False)
        ]

        while parser.parse_optional_punctuation(">") is None:
            parser.parse_shape_delimiter()
            shape.append(
                parser.parse_integer(allow_negative=False, allow_boolean=False)
            )

        return [builtin.DenseArrayBase.from_list(builtin.i64, shape)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_string("x".join(str(x) for x in self.shape.get_values()))
        printer.print_string(">")


class DomainDecompositionStrategy(ParametrizedAttribute, ABC):
    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        raise NotImplementedError("SlicingStrategy must implement calc_resize!")

    def halo_exchange_defs(self, shape: ShapeAttr) -> Iterable[ExchangeDeclarationAttr]:
        raise NotImplementedError("SlicingStrategy must implement halo_exchange_defs!")

    def comm_layout(self) -> RankTopoAttr:
        raise NotImplementedError("SlicingStrategy must implement comm_count!")


@irdl_attr_definition
class GridSlice2dAttr(DomainDecompositionStrategy):
    """
    Takes a grid with two or more dimensions, slices it along the first two into equally
    sized segments.
    """

    name = "dmp.grid_slice_2d"

    topology: RankTopoAttr

    diagonals: builtin.BoolAttr

    def __init__(self, topo: tuple[int, ...]):
        super().__init__(RankTopoAttr(topo), builtin.BoolAttr.from_int_and_width(0, 1))

    def _verify(self):
        assert len(self.topology.as_tuple()) >= 2, (
            "GridSlice2d requires at least two dimensions"
        )

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 2, "GridSlice2d requires at least two dimensions"
        for size, node_count in zip(shape, self.topology.as_tuple()):
            assert size % node_count == 0, (
                "GridSlice2d requires domain be neatly divisible by shape"
            )
        return (
            *(
                size // node_count
                for size, node_count in zip(shape, self.topology.as_tuple())
            ),
            *(size for size in shape[2:]),
        )

    def halo_exchange_defs(self, shape: ShapeAttr) -> Iterable[ExchangeDeclarationAttr]:
        yield from _flat_face_exchanges_for_dim(shape, 0)

        yield from _flat_face_exchanges_for_dim(shape, 1)

        if self.diagonals.value.data:
            raise NotImplementedError("Diagonals support not implemented yet")

    def comm_layout(self) -> RankTopoAttr:
        return RankTopoAttr(self.topology.as_tuple())


@irdl_attr_definition
class GridSlice3dAttr(DomainDecompositionStrategy):
    """
    Takes a grid with two or more dimensions, slices it along the first three.
    """

    name = "dmp.grid_slice_3d"

    topology: RankTopoAttr

    diagonals: builtin.BoolAttr

    def __init__(self, topo: tuple[int, ...]):
        super().__init__(RankTopoAttr(topo), builtin.BoolAttr.from_int_and_width(0, 1))

    def _verify(self):
        assert len(self.topology.as_tuple()) >= 3, (
            "GridSlice3d requires at least three dimensions"
        )

    def calc_resize(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 3, "GridSlice3d requires at least two dimensions"
        for size, node_count in zip(shape, self.topology.as_tuple()):
            assert size % node_count == 0, (
                "GridSlice3d requires domain be neatly divisible by shape"
            )
        return (
            *(
                size // node_count
                for size, node_count in zip(shape, self.topology.as_tuple())
            ),
            *(size for size in shape[3:]),
        )

    def halo_exchange_defs(self, shape: ShapeAttr) -> Iterable[ExchangeDeclarationAttr]:
        yield from _flat_face_exchanges_for_dim(shape, 0)

        yield from _flat_face_exchanges_for_dim(shape, 1)

        yield from _flat_face_exchanges_for_dim(shape, 2)

        if self.diagonals.value.data:
            raise NotImplementedError("Diagonals support not implemented yet")

    def comm_layout(self) -> RankTopoAttr:
        return RankTopoAttr(self.topology.as_tuple())


def _flat_face_exchanges_for_dim(
    shape: ShapeAttr, axis: int
) -> tuple[ExchangeDeclarationAttr, ...]:
    """
    Generate the two exchange delcarations to exchange the faces on the
    axis "axis".
    """
    dimensions = shape.dims
    assert axis <= dimensions

    def coords(where: Literal["start", "end"]) -> Iterable[tuple[tuple[int, int], ...]]:
        """
        Generate a series of swaps that need to be performed to exchange along "axis".

        A swap is a set of (lb,ub) tuples, one per axis of shape.

        Takes either "start" or "end" to signify if the lower (buffer start to core start) or upper
        (core end to buffer end) parts of the halo should be exchanged.

        We need to make sure that if core_size is smaller than halo size, we emit multiple exchanges.

        We need to make sure that we emit the exchanges in a way that the closest neighbor is emitted first.
        """
        # we may need to issue multiple swaps per direction, if the core size is smaller than the
        # exchanged size. This is tracked in the "slice" variable.
        slice = 0

        while True:
            swap: list[tuple[int, int]] = []
            for d in range(dimensions):
                # for the dim we want to exchange, return exchanges need to exchange either start or end
                # halo regions
                if d == axis:
                    core_size = shape.core_size(d)
                    if where == "start":
                        # where == "start" halo goes from buffer start to core start
                        # the window of data we want to send starts here
                        start = shape.buffer_start(d)
                        # calculate where the current slice starts (lowest index, no lower than start)
                        slice_start = max(
                            start, shape.core_start(d) - (core_size * (slice + 1))
                        )
                        # calculate where the current slice ends (highest index, no higher than core_start)
                        # because slice >= 0
                        slice_end = max(
                            start, shape.core_start(d) - (core_size * slice)
                        )

                        # stop swapping if swap is empty
                        if slice_end == slice_start:
                            return
                        swap.append((slice_start, slice_end))
                    else:
                        # where == "end" halo goes from core end to buffer end

                        # the window of data we want to send ends here (highest index)
                        end = shape.buffer_end(d)
                        # calculate where the current slice starts (lowest index, no lower than start)
                        # because slice >= 0, and no higher than end
                        slice_start = min(end, shape.core_end(d) + (core_size * slice))
                        # calculate where the current slice ends (highest index, no higher than core_start)
                        slice_end = min(
                            end, shape.core_end(d) + (core_size * (slice + 1))
                        )

                        # stop swapping if swap is empty
                        if slice_end == slice_start:
                            return
                        swap.append((slice_start, slice_end))

                else:
                    # for the sliced regions, "extrude" from core
                    # this way we don't exchange edges
                    swap.append((shape.core_start(d), shape.core_end(d)))

            slice += 1
            yield tuple(swap)

    return (
        # towards positive dim:
        *(
            ExchangeDeclarationAttr.from_points(
                ex1_coords,
                axis,
                dir_sign=1,
                neighbor_offset=i + 1,
            )
            for i, ex1_coords in enumerate(coords("end"))
        ),
        # towards negative dim:
        *(
            ExchangeDeclarationAttr.from_points(
                ex2_coords,
                axis,
                dir_sign=-1,
                neighbor_offset=i + 1,
            )
            for i, ex2_coords in enumerate(coords("start"))
        ),
    )


class SwapOpHasShapeInferencePatterns(HasShapeInferencePatternsTrait):
    @classmethod
    def get_shape_inference_patterns(cls):
        from xdsl.transforms.shape_inference_patterns.dmp import (
            DmpSwapShapeInference,
            DmpSwapSwapsInference,
        )

        return (DmpSwapShapeInference(), DmpSwapSwapsInference())


class SwapOpMemoryEffect(MemoryEffect):
    """
    Side effect implementation of dmp.swap.
    """

    @classmethod
    def get_effects(cls, op: Operation) -> set[EffectInstance]:
        op = cast(SwapOp, op)
        # If it's operating in value-semantic mode, it has no side effects.
        if op.swapped_values:
            return set()
        # If it's operating in reference-semantic mode, it reads and writes to its field.
        # TODO: consider the empty swaps case at some point.
        # Right now, it relies on it before inferring them, so not very safe.
        # But it could be an elegant way to generically simplify those.
        return {
            EffectInstance(MemoryEffectKind.WRITE, op.input_stencil),
            EffectInstance(MemoryEffectKind.READ, op.input_stencil),
        }


@irdl_op_definition
class SwapOp(IRDLOperation):
    """
    Declarative swap of memref regions.
    """

    name = "dmp.swap"

    input_stencil = operand_def(stencil.StencilTypeConstr)
    swapped_values = opt_result_def(stencil.TempType[Attribute])

    swaps = attr_def(builtin.ArrayAttr[ExchangeDeclarationAttr])

    strategy = attr_def(DomainDecompositionStrategy)

    traits = traits_def(SwapOpHasShapeInferencePatterns(), SwapOpMemoryEffect())

    def verify_(self) -> None:
        if self.swapped_values:
            if isinstance(self.input_stencil.type, stencil.FieldType):
                raise VerifyException(
                    "dmp.swap_op cannot have a result if input is a field"
                )
        else:
            if isinstance(self.input_stencil.type, stencil.TempType):
                raise VerifyException(
                    "dmp.swap_op must have a result if input is a temporary"
                )

    @staticmethod
    def get(
        input_stencil: SSAValue | Operation,
        strategy: DomainDecompositionStrategy,
        swaps: builtin.ArrayAttr[ExchangeDeclarationAttr] | None = None,
    ):
        input_type = SSAValue.get(input_stencil).type

        result_types = (
            input_type if isa(input_type, stencil.TempType[Attribute]) else None
        )

        if swaps is None:
            swaps = builtin.ArrayAttr[ExchangeDeclarationAttr](())

        return SwapOp.build(
            operands=[input_stencil],
            result_types=[result_types],
            attributes={
                "strategy": strategy,
                "swaps": swaps,
            },
        )


DMP = Dialect(
    "dmp",
    [
        SwapOp,
    ],
    [
        ExchangeDeclarationAttr,
        ShapeAttr,
        RankTopoAttr,
        GridSlice2dAttr,
        GridSlice3dAttr,
    ],
)
