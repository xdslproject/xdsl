from collections.abc import Sequence
from typing import ClassVar, TypeAlias

from xdsl.dialects.builtin import (
    I16,
    I64,
    ArrayAttr,
    BytesAttr,
    DenseArrayBase,
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

MeshAxesAttr: TypeAlias = DenseArrayBase[I16]


@irdl_attr_definition
class MeshAxesArrayAttr(ParametrizedAttribute, OpaqueSyntaxAttribute):
    name = "mesh.axisarray"

    axes: ArrayAttr[MeshAxesAttr]

    ELT_TYPE: ClassVar = i16

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        def parse_mesh_axes_attr():
            elements = parser.parse_comma_separated_list(
                parser.Delimiter.SQUARE,
                parser.parse_integer,
            )

            return DenseArrayBase[I16](
                cls.ELT_TYPE, BytesAttr(cls.ELT_TYPE.pack(elements))
            )

        axes = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            parse_mesh_axes_attr,
        )

        return (ArrayAttr(axes),)

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
