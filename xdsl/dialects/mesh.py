from collections.abc import Sequence
from typing import TypeAlias

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
