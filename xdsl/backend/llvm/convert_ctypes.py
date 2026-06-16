import ctypes
from collections.abc import Callable
from typing import Any

from xdsl.dialects.builtin import Float32Type, Float64Type, IntegerType, NoneType
from xdsl.dialects.llvm import LLVMPointerType, LLVMVoidType
from xdsl.ir import Attribute
from xdsl.utils.exceptions import LLVMTranslationException


class CTypeContext:
    """Registry of xDSL attribute classes to ctypes converters."""

    registry: dict[type[Attribute], Callable[[Any], Any]]
    """Map from an xDSL attribute class to a converter producing its ctypes type."""

    def __init__(self) -> None:
        self.registry = {}

    def register_ctype(
        self,
        attr_type: type[Attribute],
        converter: Callable[[Any], Any],
    ) -> None:
        self.registry[attr_type] = converter

    def to_ctype(self, type_attr: Attribute) -> Any:
        try:
            converter = self.registry[type(type_attr)]
        except KeyError:
            raise LLVMTranslationException(f"No ctypes mapping for type: {type_attr}")
        return converter(type_attr)


_INT_CTYPE_BY_WIDTH: dict[int, Any] = {
    1: ctypes.c_bool,
    8: ctypes.c_int8,
    16: ctypes.c_int16,
    32: ctypes.c_int32,
    64: ctypes.c_int64,
}


def _int_to_ctype(type_attr: IntegerType) -> Any:
    width = type_attr.width.data
    try:
        return _INT_CTYPE_BY_WIDTH[width]
    except KeyError:
        raise LLVMTranslationException(
            f"No ctypes mapping for integer of width {width}"
        )


def register_builtin_ctypes(ctx: CTypeContext) -> None:
    ctx.register_ctype(Float32Type, lambda _: ctypes.c_float)
    ctx.register_ctype(Float64Type, lambda _: ctypes.c_double)
    ctx.register_ctype(IntegerType, _int_to_ctype)
    ctx.register_ctype(LLVMPointerType, lambda _: ctypes.c_void_p)
    ctx.register_ctype(LLVMVoidType, lambda _: None)
    ctx.register_ctype(NoneType, lambda _: None)
