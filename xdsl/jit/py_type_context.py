from collections.abc import Callable
from typing import Any, NamedTuple, get_args

from typing_extensions import TypeForm

from xdsl.jit.c_type_context import CFuncSignature
from xdsl.utils.exceptions import JITException


class TypeMap(NamedTuple):
    """
    Map a Python type to a C type and optional call-boundary conversions.

    Missing converters pass values through unchanged.
    """

    python_type: type[Any]
    """Python type on the wrapped-function boundary."""

    c_type: str
    """C type used by the native function."""

    to_c: Callable[[Any], Any] | None = None
    """Convert a Python argument before the native call."""

    from_c: Callable[[Any], Any] | None = None
    """Convert the native result back to a Python value."""


class FuncTypeMap(NamedTuple):
    """Per-argument and result :class:`TypeMap` entries for a function signature."""

    arg_maps: tuple[TypeMap, ...]
    res_map: TypeMap

    def c_func_type(self) -> CFuncSignature:
        """Return the C function signature."""
        return CFuncSignature(
            tuple(type_map.c_type for type_map in self.arg_maps),
            self.res_map.c_type,
        )


class PyTypeContext:
    """
    Registry of Python types to :class:`TypeMap` entries.

    Configures native call signatures and optional value conversions when wrapping
    a :class:`RawJITFunc`. Registrations must agree with the frontend type mapping and
    with
    :class:`~xdsl.jit.c_type_context.CTypeContext` for the same logical types.
    """

    _mapping: dict[type[Any], TypeMap]

    def __init__(self):
        self._mapping = {}

    def register_type_map(self, type_map: TypeMap):
        """Register a :class:`TypeMap` for its ``python_type``."""
        self._mapping[type_map.python_type] = type_map

    def type_map(self, python_type: type[Any]) -> TypeMap:
        """Return the :class:`TypeMap` for ``python_type``."""
        try:
            return self._mapping[python_type]
        except KeyError:
            raise JITException(f"No type map for Python type: {python_type}")

    def func_type_map(self, signature: TypeForm[Callable[..., Any]]) -> FuncTypeMap:
        """Build a :class:`FuncTypeMap` from a ``Callable`` signature."""
        match get_args(signature):
            case [[*param_types], return_type]:
                return FuncTypeMap(
                    tuple(self.type_map(py_type) for py_type in param_types),
                    self.type_map(return_type),
                )
            case _:
                raise JITException(f"Unsupported signature: {signature}")
