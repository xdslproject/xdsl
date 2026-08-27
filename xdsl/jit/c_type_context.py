from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from xdsl.dialects.builtin import Float32Type, Float64Type, IntegerType, NoneType
from xdsl.ir import Attribute
from xdsl.utils.exceptions import JITException


class CFuncSignature(NamedTuple):
    """C function signature."""

    inputs: tuple[str, ...]
    output: str


class CTypeContext:
    """Registry of xDSL attribute classes to C type converters."""

    registry: dict[type[Attribute], Callable[[Any], str]]
    """Map from an xDSL attribute class to a converter producing its C type."""

    def __init__(self) -> None:
        self.registry = {}

    def register_type(
        self,
        attr_type: type[Attribute],
        converter: Callable[[Any], str],
    ) -> None:
        """Register how ``attr_type`` instances map to a C type."""
        self.registry[attr_type] = converter

    def to_c_type(self, type_attr: Attribute) -> str:
        """Return the C type for ``type_attr``."""
        try:
            converter = self.registry[type(type_attr)]
        except KeyError:
            raise JITException(f"No C type mapping for type: {type_attr}")
        return converter(type_attr)

    def to_c_func_type(
        self, inputs: Iterable[Attribute], output: Attribute
    ) -> CFuncType:
        """Build a C function signature from IR argument and result types."""
        return CFuncType(
            tuple(self.to_type(arg) for arg in inputs), self.to_type(output)
        )


_INT_TYPE_BY_WIDTH = {
    1: "_Bool",
    8: "int8_t",
    16: "int16_t",
    32: "int32_t",
    64: "int64_t",
}


def _int_to_type(type_attr: IntegerType) -> str:
    width = type_attr.width.data
    try:
        return _INT_TYPE_BY_WIDTH[width]
    except KeyError:
        raise JITException(f"No C type mapping for integer of width {width}")


def register_builtin_types(ctx: CTypeContext) -> None:
    ctx.register_type(Float32Type, lambda _: "float")
    ctx.register_type(Float64Type, lambda _: "double")
    ctx.register_type(IntegerType, _int_to_type)
    ctx.register_type(NoneType, lambda _: "void")
