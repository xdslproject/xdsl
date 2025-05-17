"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

from collections.abc import Iterable

from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerType,
    IntAttr,
    ShapedType,
)
from xdsl.ir import (
    AttributeCovT,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    ParameterDef,
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class EmitC_ArrayType(
    ParametrizedAttribute, TypeAttribute, ShapedType, ContainerType[AttributeCovT]
):
    """EmitC array type"""

    name = "emitc.array"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[AttributeCovT]

    def __init__(
        self,
        shape: Iterable[int | IntAttr],
        element_type: AttributeCovT,
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__([shape, element_type])

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("EmitC array shape must not be empty")

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            shape, type = parser.parse_ranked_shape()
            if not shape:
                raise parser.raise_error("EmitC array shape must not be empty")
            return ArrayAttr(IntAttr(dim) for dim in shape), type

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(
                self.shape, lambda dim: printer.print_string(f"{dim.data}"), "x"
            )
            printer.print_string("x")
            printer.print_attribute(self.element_type)


EmitC = Dialect(
    "emitc",
    [],
    [
        EmitC_ArrayType,
    ],
)
