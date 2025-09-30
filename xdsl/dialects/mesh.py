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
    """
    MeshAxesArrayAttr attribute for representing mutiple mesh axes.

    See [external documentation](https://mlir.llvm.org/docs/Dialects/Shard/#gridaxesarrayattr).
    """

    name = "mesh.axisarray"

    axes: ArrayAttr[MeshAxesAttr]

    ELT_TYPE: ClassVar = i16

    @classmethod
    def parse_mesh_axes_attr(cls, parser: AttrParser) -> DenseArrayBase[I16]:
        """
        Parses a single MeshAxesAttr, e.g. [1, 4, 7, 8]
        """
        elements = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            parser.parse_integer,
        )

        return DenseArrayBase[I16](cls.ELT_TYPE, BytesAttr(cls.ELT_TYPE.pack(elements)))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """
        Parses a MeshAxesArrayAttr, which has the syntax of a list
        of lists, e.g.:

        [[1, 2, 3], [], [4, 5]]
        """
        axes = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE,
            lambda: cls.parse_mesh_axes_attr(parser),
        )

        return (ArrayAttr(axes),)

    @classmethod
    def print_sublist(cls, sublist: MeshAxesAttr, printer: Printer) -> None:
        """
        Prints a single MeshAxesAttr, e.g. [1, 4, 6, 8]
        """
        with printer.in_square_brackets():
            printer.print_list(sublist.get_values(), printer.print_int)

    def print_parameters(self, printer: Printer) -> None:
        """
        Prints a MeshAxesArrayAttr, which has the syntax of a list
        of lists, e.g.:

        [[1, 2, 3], [], [4, 5]]
        """
        with printer.in_square_brackets():
            printer.print_list(
                self.axes.data,
                lambda x: MeshAxesArrayAttr.print_sublist(x, printer),
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
