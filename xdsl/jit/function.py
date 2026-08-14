from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, Protocol, cast, overload

from typing_extensions import TypeForm, TypeVar

from xdsl.jit.py_type_context import PyTypeContext


class CFunc(Protocol):
    """A callable native function pointer."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class CFuncType(Protocol):
    """A ctypes function-pointer type."""

    @overload
    def __call__(self) -> CFunc: ...

    @overload
    def __call__(self, address: int, /) -> CFunc: ...

    @overload
    def __call__(self, func: Callable[..., Any], /) -> CFunc: ...


@dataclass(slots=True)
class RawJITFunc:
    """
    A jitted function exposed as a ctypes callable.

    Backends may subclass this to retain native runtime state that must outlive
    calls through ``c_func``.
    """

    c_func_type: CFuncType
    """The ``CFUNCTYPE`` describing the native calling convention."""

    c_func: CFunc
    """Bound ctypes function object for the native entry point."""


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class WrappedJITFunc(Generic[P, R]):
    """
    A Python-callable wrapper around a :class:`RawJITFunc`.

    Invoking the instance marshals arguments to ctypes, calls the native function,
    and converts the result back to a Python value.
    """

    raw_func: RawJITFunc
    """Underlying ctypes binding."""

    original_func: Callable[P, R]
    """The undecorated Python function."""

    __call__: Callable[P, R]
    """Marshaling entry point for calls."""


def wrap_jit_func(
    raw_func: RawJITFunc,
    original_func: Callable[P, R],
    signature: TypeForm[Callable[P, R]],
    py_type_context: PyTypeContext,
) -> WrappedJITFunc[P, R]:
    """
    Wrap a :class:`RawJITFunc` as a :class:`WrappedJITFunc`.

    Builds argument/result converters from ``signature`` and checks that the
    resulting ``CFUNCTYPE`` matches ``raw_func.c_func_type``.
    """
    func_type_map = py_type_context.func_type_map(signature)
    assert raw_func.c_func_type == func_type_map.c_func_type(), (
        f"CTypes signature from IR ({raw_func.c_func_type}) does not "
        f"match signature from Python TypeMaps ({func_type_map.c_func_type()})."
    )

    def fn(*args: Any) -> R:
        ctype_args = tuple(
            m.to_ctype(a) for m, a in zip(func_type_map.arg_maps, args, strict=True)
        )
        ctype_res = raw_func.c_func(*ctype_args)
        return func_type_map.res_map.from_ctype(ctype_res)

    return WrappedJITFunc(raw_func, original_func, cast(Callable[P, R], fn))
