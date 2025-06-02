"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

from collections.abc import Iterable

from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    ContainerType,
    Float16Type,
    Float32Type,
    Float64Type,
    IndexType,
    IntAttr,
    IntegerType,
    ShapedType,
)
from xdsl.ir import (
    Attribute,
    AttributeCovT,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import ParameterDef, irdl_attr_definition, isa
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

        if isinstance(element_type, EmitC_ArrayType):
            raise VerifyException(
                "EmitC array element type cannot be another EmitC_ArrayType."
            )

        # Check that the element type is a supported EmitC type.
        if not self._is_valid_element_type(element_type):
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
            return ArrayAttr(IntAttr(dim) for dim in shape), type

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(
                self.shape, lambda dim: printer.print_string(f"{dim.data}"), "x"
            )
            printer.print_string("x")
            printer.print_attribute(self.element_type)

    def _is_valid_element_type(self, element_type: Attribute) -> bool:
        """
        Check if the element type is valid for EmitC_ArrayType.
        See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L77).
        """
        return is_integer_index_or_opaque_type(element_type) or is_supported_float_type(
            element_type
        )


_SUPPORTED_BITWIDTHS = (1, 8, 16, 32, 64)


def _is_supported_integer_type(type_attr: Attribute) -> bool:
    """
    Check if an IntegerType is supported by EmitC.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
    """
    return (
        isinstance(type_attr, IntegerType)
        and type_attr.width.data in _SUPPORTED_BITWIDTHS
    )


def is_supported_float_type(type_attr: Attribute) -> bool:
    """
    Check if a type is a supported floating-point type in EmitC.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L117)
    """
    match type_attr:
        case Float16Type() | BFloat16Type() | Float32Type() | Float64Type():
            return True
        case _:
            return False


def is_integer_index_or_opaque_type(
    type_attr: Attribute,
) -> bool:
    """
    Check if a type is an integer, index, or opaque type.

    The emitC opaque type is not implemented yet so this function currently checks
    only for integer and index types.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L112).
    """
    return _is_supported_integer_type(type_attr) or isa(type_attr, IndexType)


EmitC = Dialect(
    "emitc",
    [],
    [
        EmitC_ArrayType,
    ],
)
