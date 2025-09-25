from __future__ import annotations

from collections.abc import Sequence
from enum import auto
from typing import TypeAlias

from xdsl.dialects.builtin import (
    I16,
    I64,
    ArrayAttr,
    BytesAttr,
    DenseArrayBase,
    FlatSymbolRefAttr,
    SymbolNameConstraint,
    i64,
)
from xdsl.dialects.utils.dimension_list import DimensionList
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Pure, SymbolOpInterface
from xdsl.utils.str_enum import StrEnum

MeshAxesAttr: TypeAlias = DenseArrayBase[I16]


@irdl_attr_definition
class MeshAxesArrayAttr(ParametrizedAttribute):
    name = "mesh.axisarray"

    axes: ArrayAttr[MeshAxesAttr]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_square_brackets():
            printer.print_attribute(self.axes)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        axes = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            parser.parse_attribute,
        )

        return axes


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
class ReductionKindAttr(EnumAttribute[ReductionKind]):
    name = "mesh.partial"

    assembly_format = "$value"

    @classmethod
    def sum(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.SUM)

    @classmethod
    def max(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.MAX)

    @classmethod
    def min(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.MIN)

    @classmethod
    def product(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.PRODUCT)

    @classmethod
    def average(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.AVERAGE)

    @classmethod
    def bitwise_and(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.BITWISE_AND)

    @classmethod
    def bitwise_or(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.BITWISE_OR)

    @classmethod
    def bitwise_xor(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.BITWISE_XOR)

    @classmethod
    def generic(cls) -> ReductionKindAttr:
        return ReductionKindAttr(ReductionKind.GENERIC)


@irdl_attr_definition
class ShardingType(ParametrizedAttribute, TypeAttribute):
    name = "mesh.sharding"


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

    # TODO: Implement assembly_format
    # custom_directives = (DynamicIndexList,)


Mesh = Dialect(
    "mesh",
    [
        MeshOp,
        ShardingOp,
    ],
    [
        ReductionKindAttr,
        ShardingType,
        MeshAxesArrayAttr,
    ],
)
