from abc import ABC
from collections.abc import Sequence
from enum import auto
from typing import Self, TypeAlias

from xdsl.dialects.builtin import (
    I16,
    I64,
    ArrayAttr,
    BytesAttr,
    DenseArrayBase,
    FlatSymbolRefAttr,
    IndexType,
    IndexTypeConstr,
    IntegerAttr,
    SymbolNameConstraint,
    TensorType,
    UnitAttr,
    i16,
    i64,
)
from xdsl.dialects.utils import DimensionList, DynamicIndexList
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    OpaqueSyntaxAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import Pure, SymbolOpInterface
from xdsl.utils.str_enum import StrEnum

################################################################################
# Types and attributes                                                         #
################################################################################

MeshAxis: TypeAlias = I16
"""
The type used to represent numbers on a mesh axis.

See [the MLIR definition](https://github.com/llvm/llvm-project/blob/6146a88f60492b520a36f8f8f3231e15f3cc6082/mlir/include/mlir/Dialect/Mesh/IR/MeshBase.td#L39).
"""


MeshAxesAttr: TypeAlias = DenseArrayBase[MeshAxis]
"""
The type used to represent a list of mesh axes.

See [the MLIR definition](https://github.com/llvm/llvm-project/blob/6146a88f60492b520a36f8f8f3231e15f3cc6082/mlir/include/mlir/Dialect/Mesh/IR/MeshBase.td#L40).
"""


def _parse_mesh_axes_attr(parser: AttrParser) -> MeshAxesAttr:
    """
    Parses a single MeshAxesAttr, e.g. [1, 4, 7, 8]
    """
    elements = parser.parse_comma_separated_list(
        parser.Delimiter.SQUARE,
        parser.parse_integer,
    )

    return MeshAxesAttr(i16, BytesAttr(i16.pack(elements)))


def _print_sublist(printer: Printer, sublist: MeshAxesAttr) -> None:
    """
    Prints a single MeshAxesAttr, e.g. [1, 4, 6, 8]
    """
    with printer.in_square_brackets():
        printer.print_list(sublist.get_values(), printer.print_int)


@irdl_attr_definition
class MeshAxesArrayAttr(ParametrizedAttribute, OpaqueSyntaxAttribute):
    """
    MeshAxesArrayAttr attribute for representing mutiple mesh axes.

    Reflects [the MLIR attribute](https://github.com/llvm/llvm-project/blob/6146a88f60492b520a36f8f8f3231e15f3cc6082/mlir/include/mlir/Dialect/Mesh/IR/MeshBase.td#L83).
    """

    name = "mesh.axisarray"

    axes: ArrayAttr[MeshAxesAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """
        Parses a MeshAxesArrayAttr, which has the syntax of a list
        of lists, e.g.:

        [[1, 2, 3], [], [4, 5]]
        """
        axes = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            lambda: _parse_mesh_axes_attr(parser),
        )

        return (ArrayAttr(axes),)

    def print_parameters(self, printer: Printer) -> None:
        """
        Prints a MeshAxesArrayAttr, which has the syntax of a list
        of lists, e.g.:

        [[1, 2, 3], [], [4, 5]]
        """
        with printer.in_square_brackets():
            printer.print_list(
                self.axes.data,
                lambda x: _print_sublist(printer, x),
            )


class ReductionKind(StrEnum):
    "Reduction kind for mesh dialect"

    SUM = auto()
    MAX = auto()
    MIN = auto()
    PRODUCT = auto()
    AVERAGE = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    GENERIC = auto()


@irdl_attr_definition
class ReductionKindAttr(EnumAttribute[ReductionKind], SpacedOpaqueSyntaxAttribute):
    name = "mesh.partial"

    assembly_format = "$value"


@irdl_attr_definition
class ShardingType(ParametrizedAttribute, TypeAttribute):
    name = "mesh.sharding"


################################################################################
# Collevtive communication ops                                                 #
################################################################################


class CollectiveCommunicationOp(IRDLOperation, ABC):
    """
    Base class for collective communication ops.
    """

    mesh = prop_def(FlatSymbolRefAttr)
    mesh_axes = prop_def(MeshAxesAttr, default_value=MeshAxesAttr(i16, BytesAttr(b"")))


@irdl_op_definition
class BroadcastOp(CollectiveCommunicationOp):
    """
    Broadcast tensor from one device to many devices.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardbroadcast-shardbroadcastop).
    """

    name = "mesh.broadcast"

    input = operand_def(TensorType)
    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class GatherOp(CollectiveCommunicationOp):
    """
    Gather tensor shards from many devices to a single device.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardgather-shardgatherop).
    """

    name = "mesh.gather"

    input = operand_def(TensorType)
    gather_axis = prop_def(IntegerAttr.constr(IndexTypeConstr))
    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? "
        + "`gather_axis` `=` $gather_axis "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ScatterOp(CollectiveCommunicationOp):
    """
    Scatter tensor over a device mesh.

    For each device group split the input tensor on the `root` device along
    axis `scatter_axis` and scatter the parts across the group devices.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardscatter-shardscatterop).
    """

    name = "mesh.scatter"

    input = operand_def(TensorType)
    scatter_axis = prop_def(IntegerAttr.constr(IndexTypeConstr))

    root = prop_def(DenseArrayBase[I64])
    root_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    traits = traits_def(
        Pure(),
    )

    assembly_format = (
        "$input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? "
        + "`scatter_axis` `=` $scatter_axis "
        + "`root` `=` custom<DynamicIndexList>($root_dynamic, $root) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class RecvOp(CollectiveCommunicationOp):
    """
    Receive from a device within a device group.
    """

    name = "mesh.recv"

    input = operand_def(TensorType)
    source = opt_prop_def(DenseArrayBase[I64])
    source_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    assembly_format = (
        "$input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? "
        + "(`source` `=` custom<DynamicIndexList>($source_dynamic, $source)^)? "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class SendOp(CollectiveCommunicationOp):
    """
    Send from one device to another within a device group.
    """

    name = "mesh.send"

    input = operand_def(TensorType)

    destination = prop_def(DenseArrayBase[I64])
    destination_dynamic = var_operand_def(IndexType)

    result = result_def(TensorType)

    assembly_format = (
        "$input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? "
        + "`destination` `=` custom<DynamicIndexList>($destination_dynamic, $destination) "
        + "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (DynamicIndexList,)


@irdl_op_definition
class ShiftOp(CollectiveCommunicationOp):
    """
    Shift over a device mesh.

    Within each device group shift along `shift_axis` by `offset`. If the
    `rotate` flag is present a rotation is performed instead of a shift.
    """

    name = "mesh.shift"

    input = operand_def(TensorType)

    shift_axis = prop_def(IntegerAttr.constr(IndexTypeConstr))
    offset = prop_def(IntegerAttr[I64])
    rotate = prop_def(UnitAttr)

    result = result_def(TensorType)

    traits = traits_def(
        Pure(),
    )

    assembly_format = (
        "$input `on` $mesh (`mesh_axes` `=` $mesh_axes^)? "
        + "`shift_axis` `=` $shift_axis "
        + "`offset` `=` $offset "
        + "(`rotate` $rotate^)? "
        + "attr-dict `:` type($input) `->` type($result)"
    )


################################################################################
# Operations on mesh                                                           #
################################################################################


@irdl_op_definition
class MeshOp(IRDLOperation):
    name = "mesh.mesh"

    sym_name = prop_def(SymbolNameConstraint())
    shape = prop_def(DenseArrayBase[I64])

    traits = traits_def(SymbolOpInterface())

    assembly_format = (
        "$sym_name `(` `shape` `=` custom<DimensionList>($shape) `)` attr-dict"
    )

    custom_directives = (DimensionList,)

    def verify_(self):
        if not self.shape.get_values():
            raise VerifyException(
                "'mesh.mesh' op rank of mesh is expected to be a positive integer"
            )


################################################################################
# Sharding operations                                                          #
################################################################################


@irdl_op_definition
class ShardingOp(IRDLOperation):
    """
    Mesh dialect sharding operation.

    Note: `halo_sizes` and `sharded_dims_offsets` are mutually exlcusive.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardsharding-shardshardingop)
    """

    name = "mesh.sharding"

    mesh = prop_def(FlatSymbolRefAttr)
    split_axes = prop_def(MeshAxesArrayAttr)
    partial_axes = opt_prop_def(MeshAxesAttr)
    partial_type = opt_prop_def(ReductionKindAttr)
    static_sharded_dims_offsets = prop_def(
        DenseArrayBase[I64], default_value=DenseArrayBase[I64](i64, BytesAttr(b""))
    )
    dynamic_sharded_dims_offsets = var_operand_def(I64)
    static_halo_sizes = prop_def(
        DenseArrayBase[I64], default_value=DenseArrayBase[I64](i64, BytesAttr(b""))
    )
    dynamic_halo_sizes = var_operand_def(I64)

    result = result_def(ShardingType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(
        Pure(),
    )

    assembly_format = (
        "$mesh `split_axes` "
        + "`=` $split_axes (`partial` `=` $partial_type $partial_axes^)? "
        + "(`halo_sizes` `=` custom<DynamicIndexList>($dynamic_halo_sizes, $static_halo_sizes)^)? "
        + "(`sharded_dims_offsets` `=` "
        + "custom<DynamicIndexList>($dynamic_sharded_dims_offsets, $static_sharded_dims_offsets)^)? "
        + "attr-dict `:` type($result)"
    )

    custom_directives = (DynamicIndexList,)

    def verify_(self) -> None:
        dims_offsets = (
            self.static_sharded_dims_offsets or self.dynamic_sharded_dims_offsets
        )
        halo_sizes = self.static_halo_sizes or self.dynamic_halo_sizes

        if dims_offsets and halo_sizes:
            raise VerifyException(
                "'mesh.sharding' cannot use both `halo_sizes` and `sharded_dims_offsets`"
            )


@irdl_op_definition
class ShardOp(IRDLOperation):
    """
    Annotate on how a tensor is sharded across a shard.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#shardshard-shardshardop).
    """

    name = "mesh.shard"

    src = operand_def(TensorType)
    sharding = operand_def(ShardingType)
    annotate_for_users = opt_prop_def(UnitAttr)

    result = result_def(TensorType)

    traits = traits_def(
        Pure(),
    )

    def __init__(
        self,
        src: SSAValue,
        sharding: SSAValue,
        annotate_for_users: UnitAttr | None,
    ):
        return super().__init__(
            operands=[src, sharding],
            result_types=[src.type],
            properties={
                "annotate_for_users": annotate_for_users,
            },
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.src)
        printer.print_string(" to ")
        printer.print_ssa_value(self.sharding)

        if self.annotate_for_users:
            printer.print_string(" annotate_for_users")

        printer.print_string(" : ")
        printer.print_attribute(self.src.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        tensor = parser.parse_operand()
        parser.parse_keyword("to")
        sharding = parser.parse_operand()

        annotate_for_users = None
        if parser.parse_optional_keyword("annotate_for_users"):
            annotate_for_users = UnitAttr()

        parser.parse_punctuation(":")
        parser.parse_type()

        return cls(tensor, sharding, annotate_for_users)


Mesh = Dialect(
    "mesh",
    [
        BroadcastOp,
        GatherOp,
        RecvOp,
        SendOp,
        ScatterOp,
        ShiftOp,
        MeshOp,
        ShardingOp,
        ShardOp,
    ],
    [
        ReductionKindAttr,
        ShardingType,
        MeshAxesArrayAttr,
    ],
)
