from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, Protocol, cast

from typing_extensions import TypeForm, TypeVar

from xdsl.jit.c_type_context import CFuncSignature
from xdsl.jit.py_type_context import PyTypeContext
from xdsl.utils.exceptions import JITException


class CFunc(Protocol):
    """A callable native function pointer."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class RawJITFunc:
    """
    A JIT-compiled native function.

    Backends may subclass this to retain native runtime state that must outlive
    calls through ``c_func``.
    """

    c_func_type: CFuncSignature
    """Native function signature."""

    c_func: CFunc
    """Entry point enforcing ``c_func_type``, valid while this object remains alive."""


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class WrappedJITFunc(Generic[P, R]):
    """
    A Python-callable wrapper around a :class:`RawJITFunc`.

    Applies configured argument and result conversions around the native call.
    """

    raw_func: RawJITFunc
    """Underlying native binding."""

    original_func: Callable[P, R]
    """The undecorated Python function."""

    __call__: Callable[P, R]
    """Wrapped call implementation."""


def wrap_jit_func(
    raw_func: RawJITFunc,
    original_func: Callable[P, R],
    signature: TypeForm[Callable[P, R]],
    py_type_context: PyTypeContext,
) -> WrappedJITFunc[P, R]:
    """
    Wrap a :class:`RawJITFunc` as a :class:`WrappedJITFunc`.

    Builds argument/result converters from ``signature`` and checks that the
    resulting C signature matches ``raw_func.c_func_type``.
    """
    func_type_map = py_type_context.func_type_map(signature)
    expected_c_func_type = func_type_map.c_func_type()
    if raw_func.c_func_type != expected_c_func_type:
        raise JITException(
            f"C signature from IR ({raw_func.c_func_type}) does not "
            f"match signature from Python type maps ({expected_c_func_type})."
        )

    arg_converters = tuple(type_map.to_c for type_map in func_type_map.arg_maps)
    result_converter = func_type_map.res_map.from_c
    c_func = raw_func.c_func
    if result_converter is None and all(
        converter is None for converter in arg_converters
    ):
        return WrappedJITFunc(raw_func, original_func, cast(Callable[P, R], c_func))

    arg_count = len(arg_converters)

    def fn(*args: P.args, **kwargs: P.kwargs) -> R:
        if kwargs:
            raise JITException("JIT functions do not support keyword arguments.")
        if len(args) != arg_count:
            raise TypeError(
                f"JIT function expects {arg_count} arguments, got {len(args)}"
            )
        call_args = tuple(
            arg if converter is None else converter(arg)
            for converter, arg in zip(arg_converters, args)
        )
        result = c_func(*call_args)
        if result_converter is not None:
            result = result_converter(result)
        return result

    return WrappedJITFunc(raw_func, original_func, fn)
