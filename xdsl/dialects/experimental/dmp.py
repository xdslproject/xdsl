"""
A dialect for handling distributed memory parallelism (DMP).

This is xDSL only for now.

This dialect aims to provide the tools necessary to facilitate the creation
and lowering of stencil (and other) computations in a manner that
makes them run on node clusters.
"""
from __future__ import annotations

from math import prod
from typing import Sequence

from xdsl.dialects import builtin, memref, stencil
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    ParameterDef,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.hints import isa

# helpers for named dimensions:
DIM_X = 0
DIM_Y = 1
DIM_Z = 2


@irdl_attr_definition
class HaloExchangeDecl(ParametrizedAttribute):
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

    name = "dmp.exchange_decl"

    offset_: ParameterDef[builtin.DenseArrayBase]
    size_: ParameterDef[builtin.DenseArrayBase]
    source_offset_: ParameterDef[builtin.DenseArrayBase]
    neighbor_: ParameterDef[builtin.DenseArrayBase]

    def __init__(
        self,
        offset: Sequence[int],
        size: Sequence[int],
        source_offset: Sequence[int],
        neighbor: Sequence[int],
    ):
        data_type = builtin.i64
        super().__init__(
            [
                builtin.DenseArrayBase.from_list(data_type, offset),
                builtin.DenseArrayBase.from_list(data_type, size),
                builtin.DenseArrayBase.from_list(data_type, source_offset),
                builtin.DenseArrayBase.from_list(data_type, neighbor),
            ]
        )

    @property
    def offset(self) -> tuple[int, ...]:
        data = self.offset_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def size(self) -> tuple[int, ...]:
        data = self.size_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def source_offset(self) -> tuple[int, ...]:
        data = self.source_offset_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def neighbor(self) -> tuple[int, ...]:
        data = self.neighbor_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def elem_count(self) -> int:
        return prod(self.size)

    @property
    def dim(self) -> int:
        return len(self.size)

    def source_area(self) -> HaloExchangeDecl:
        """
        Since a HaloExchangeDef by default specifies the area to receive into,
        this method returns the area that should be read from.
        """
        # we set source_offset to all zero, so that repeated calls to source_area never
        # return the dest area
        return HaloExchangeDecl(
            offset=tuple(
                val + offs for val, offs in zip(self.offset, self.source_offset)
            ),
            size=self.size,
            source_offset=tuple(0 for _ in range(len(self.source_offset))),
            neighbor=self.neighbor,
        )

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<at [")
        printer.print_list(self.offset, lambda x: printer.print_string(str(x)))
        printer.print_string("] size [")
        printer.print_list(self.size, lambda x: printer.print_string(str(x)))
        printer.print_string("] source offset [")
        printer.print_list(self.source_offset, lambda x: printer.print_string(str(x)))
        printer.print_string(f"] to {list(self.neighbor)}>")

    # TODO: def parse_parameters()


@irdl_attr_definition
class HaloShapeInformation(ParametrizedAttribute):
    """
    This represents shape information that is attached to halo operations.

    On the terminology used:

    In each dimension, we are given four points. We abbreviate them in
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

    name = "dmp.shape_with_halo"

    buff_lb_: ParameterDef[builtin.DenseArrayBase]
    buff_ub_: ParameterDef[builtin.DenseArrayBase]
    core_lb_: ParameterDef[builtin.DenseArrayBase]
    core_ub_: ParameterDef[builtin.DenseArrayBase]

    @property
    def buff_lb(self) -> tuple[int, ...]:
        data = self.buff_lb_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def buff_ub(self) -> tuple[int, ...]:
        data = self.buff_ub_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def core_lb(self) -> tuple[int, ...]:
        data = self.core_lb_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def core_ub(self) -> tuple[int, ...]:
        data = self.core_ub_.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    @property
    def dims(self) -> int:
        """
        Number of axis of the data (len(shape))
        """
        return len(self.core_ub)

    @staticmethod
    def from_index_attrs(
        buff_lb: stencil.IndexAttr,
        core_lb: stencil.IndexAttr,
        core_ub: stencil.IndexAttr,
        buff_ub: stencil.IndexAttr,
    ):
        data_type = builtin.i64
        return HaloShapeInformation(
            [
                builtin.DenseArrayBase.from_list(data_type, tuple(data))
                for data in (buff_lb, buff_ub, core_lb, core_ub)
            ]
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
class NodeGrid(ParametrizedAttribute):
    """
    This attribute specifies the node layout used to distribute the computation.

    dmp.grid<3x3> means nine nodes organized in a 3x3 grid.

    This allows for higher-dimensional grids as well, e.g. dmp.grid<3x3x3> for
    3-dimensional data.
    """

    name = "dmp.grid"

    shape: ParameterDef[builtin.DenseArrayBase]

    def __init__(self, shape: Sequence[int]):
        if len(shape) < 1:
            raise ValueError("dmp.grid must have at least one dimension!")
        super().__init__([builtin.DenseArrayBase.from_list(builtin.i64, shape)])

    def as_tuple(self) -> tuple[int, ...]:
        shape = self.shape.as_tuple()
        assert isa(shape, tuple[int, ...])
        return shape

    def node_count(self) -> int:
        return prod(self.as_tuple())

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")

        shape: list[int] = [parser.parse_integer(allow_negative=False)]

        while parser.parse_optional_characters("x") is not None:
            shape.append(parser.parse_integer(allow_negative=False))

        parser.parse_characters(">")

        return [builtin.DenseArrayBase.from_list(builtin.i64, shape)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_string("x".join(str(x) for x in self.shape.as_tuple()))
        printer.print_string(">")


@irdl_op_definition
class HaloSwapOp(IRDLOperation):
    """
    Declarative swap of memref regions.
    """

    name = "dmp.swap"

    input_stencil: Operand = operand_def(
        stencil.TempType[Attribute] | memref.MemRefType[Attribute]
    )

    # shape: HaloShapeInformation| None = opt_attr_def(HaloShapeInformation)
    swaps: builtin.ArrayAttr[HaloExchangeDecl] | None = opt_attr_def(
        builtin.ArrayAttr[HaloExchangeDecl]
    )
    nodes: NodeGrid | None = opt_attr_def(NodeGrid)

    @staticmethod
    def get(input_stencil: SSAValue | Operation):
        return HaloSwapOp.build(operands=[input_stencil])


@irdl_op_definition
class GatherOp(IRDLOperation):
    """
    Gather a scattered array back to one node
    """

    name = "dmp.gather"

    local_field: Operand = operand_def(memref.MemRefType)

    my_rank: Operand = operand_def(builtin.IndexType)

    root_rank: builtin.IntegerAttr[builtin.IntegerType] = attr_def(
        builtin.IntegerAttr[builtin.IntegerType]
    )

    global_shape: HaloShapeInformation = attr_def(HaloShapeInformation)

    when_root_block: Region = region_def("single_block")
    """
    Contains code to be executed as root rank
    """

    retain_order: builtin.UnitAttr | None = opt_attr_def(builtin.UnitAttr)
    """
    A normal mpi.gather() will result in a reordering of the data, where each
    nodes data will be placed sequentially into the buffer, without any
    knowledge of the node layout.

    Given a decomposition like this (number on the grid cell is the node id):

    1 1 2 2 3 3
    1 1 2 2 3 3
    4 4 5 5 6 6
    4 4 5 5 6 6

    The mpi.gather will result in the following layout in the buffer:

    1 1 1 1 2 2
    2 2 3 3 3 3
    4 4 4 4 5 5
    5 5 6 6 6 6

    If retain_order is set, the gather op will make sure that the data in the
    output memred retains the same order as it is "logically":

    1 1 2 2 3 3
    1 1 2 2 3 3
    4 4 5 5 6 6
    4 4 5 5 6 6
    """

    # TODO: implement

    # TODO: fix __init__
    def __init__(
        self,
        local_field: SSAValue | Operation,
        root_rank: int = 0,
        retain_order: bool = True,
    ):
        attrs: dict[str, Attribute] = {
            "root_rank": builtin.IntAttr(root_rank),
        }
        if retain_order:
            attrs["retain_order"] = builtin.UnitAttr()

        super().__init__(operands=[local_field], attributes=attrs)


@irdl_op_definition
class ScatterOp(IRDLOperation):
    name = "dmp.scatter"

    global_field: Operand = operand_def(memref.MemRefType)

    my_rank: Operand = operand_def(builtin.IndexType)

    global_shape: HaloShapeInformation = attr_def(HaloShapeInformation)

    def __init__(self, ref: SSAValue | Operand, shape: HaloShapeInformation):
        super().__init__(operands=[ref], attributes={"global_shape": shape})


DMP = Dialect(
    [
        HaloSwapOp,
        GatherOp,
        ScatterOp,
    ],
    [
        HaloExchangeDecl,
        HaloShapeInformation,
        NodeGrid,
    ],
)
