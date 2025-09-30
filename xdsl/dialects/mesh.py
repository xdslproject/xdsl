from collections.abc import Sequence
from typing import TypeAlias

from xdsl.dialects.builtin import (
    I16,
    I64,
    ArrayAttr,
    DenseArrayBase,
    IntegerAttr,
    SymbolNameConstraint,
    i16,
)
from xdsl.dialects.utils.dimension_list import DimensionList
from xdsl.ir import (
    Attribute,
    Dialect,
    OpaqueSyntaxAttribute,
    ParametrizedAttribute,
    VerifyException,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import SymbolOpInterface
from xdsl.utils.hints import isa

MeshAxesAttr: TypeAlias = DenseArrayBase[I16]


@irdl_attr_definition
class MeshAxesArrayAttr(ParametrizedAttribute, OpaqueSyntaxAttribute):
    name = "mesh.axisarray"

    axes: ArrayAttr[MeshAxesAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        axes = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            parser.parse_attribute,  # TODO: Update to use parse_dense_i16
        )

        assert isa(axes, list[ArrayAttr[IntegerAttr[I64]]])

        axes_i16: list[MeshAxesAttr] = []

        for array_attr in axes:
            dense = DenseArrayBase[I16].from_list(
                i16, list(map(lambda attr: attr.value.data, array_attr.data))
            )
            axes_i16.append(dense)

        return (ArrayAttr(axes_i16),)

    def print_parameters(self, printer: Printer) -> None:
        def print_sublist(sublist: MeshAxesAttr):
            with printer.in_square_brackets():
                printer.print_list(sublist.get_values(), printer.print_int)

        with printer.in_square_brackets():
            printer.print_list(
                self.axes.data,
                print_sublist,
            )


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


Mesh = Dialect(
    "mesh",
    [
        MeshOp,
    ],
    [
        MeshAxesArrayAttr,
    ],
)
