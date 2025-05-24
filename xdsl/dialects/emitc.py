"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

from collections.abc import Iterable

from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    BFloat16Type,
    ContainerType,
    Float16Type,
    IndexType,
    IntAttr,
    IntegerType,
    ShapedType,
    TensorType,
    TupleType,
)
from xdsl.ir import (
    Attribute,
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

        for dim_attr in self.shape.data:
            if dim_attr.data < 0:
                raise VerifyException(
                    "EmitC array dimensions must have non-negative size"
                )

        element_type = self.get_element_type()

        # Check that the element type is not an ArrayType itself.
        if isinstance(element_type, EmitC_ArrayType):
            raise VerifyException(
                "EmitC array element type cannot be another EmitC_ArrayType."
            )

        # Check that the element type is a supported EmitC type.
        if not is_supported_emitc_type(element_type):
            raise VerifyException(
                f"EmitC array element type '{element_type}' is not a supported EmitC type."
            )

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
            if isinstance(type, EmitC_ArrayType):
                raise parser.raise_error(
                    "EmitC array element type cannot be another EmitC_ArrayType."
                )
            if not is_supported_emitc_type(type):
                raise parser.raise_error(f"invalid array element type '{type}'")
            return ArrayAttr(IntAttr(dim) for dim in shape), type

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(
                self.shape, lambda dim: printer.print_string(f"{dim.data}"), "x"
            )
            printer.print_string("x")
            printer.print_attribute(self.element_type)


def is_supported_integer_type(type_attr: Attribute) -> bool:
    """Check if an IntegerType is supported by EmitC."""
    if isinstance(type_attr, IntegerType):
        return type_attr.width.data in [1, 8, 16, 32, 64]
    return False


def is_supported_float_type(type_attr: Attribute) -> bool:
    """Check if a FloatType is supported by EmitC."""
    if isinstance(type_attr, Float16Type | BFloat16Type):
        return True
    if isinstance(type_attr, AnyFloat):
        return type_attr.bitwidth in [32, 64]
    return False


def is_supported_emitc_type(type_attr: Attribute) -> bool:
    """Check if a type is supported by EmitC."""

    if is_supported_float_type(type_attr):
        return True

    if isinstance(type_attr, IntegerType):
        # is_supported_integer_type handles the width check
        if is_supported_integer_type(type_attr):
            return True

    if isinstance(type_attr, IndexType):
        return True

    if isinstance(type_attr, EmitC_ArrayType):
        elem_type: Attribute = type_attr.get_element_type()
        return not isinstance(elem_type, EmitC_ArrayType) and is_supported_emitc_type(
            elem_type
        )

    if isinstance(type_attr, AnyFloat):
        return is_supported_float_type(type_attr)

    if isinstance(type_attr, TensorType):
        elem_type = type_attr.get_element_type()
        if isinstance(elem_type, EmitC_ArrayType):
            return False
        return is_supported_emitc_type(elem_type)

    if isinstance(type_attr, TupleType):
        return all(
            not isinstance(t, EmitC_ArrayType) and is_supported_emitc_type(t)
            for t in type_attr.types
        )

    return False


EmitC = Dialect(
    "emitc",
    [],
    [
        EmitC_ArrayType,
    ],
)
