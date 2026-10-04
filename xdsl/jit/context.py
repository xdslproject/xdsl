import abc
from collections.abc import Callable
from typing import ParamSpec

from typing_extensions import TypeForm, TypeVar

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.jit.c_type_context import CTypeContext
from xdsl.jit.function import RawJITFunc, WrappedJITFunc, wrap_jit_func
from xdsl.jit.py_type_context import PyTypeContext, TypeMap

P = ParamSpec("P")
R = TypeVar("R")


class JITBackend(abc.ABC):
    """
    Compile a module symbol to a :class:`~xdsl.jit.function.RawJITFunc`.

    Implementations receive the module produced by the frontend, lower it to the
    backend’s dialect, and bind ``symbol`` for native calls.
    """

    c_type_context: CTypeContext
    """IR attribute to C type registry."""

    def __init__(self):
        """Initialize an empty :class:`~xdsl.jit.c_type_context.CTypeContext`."""
        super().__init__()
        self.c_type_context = CTypeContext()

    @abc.abstractmethod
    def jit(
        self,
        mlir_module: builtin.ModuleOp,
        symbol: str,
        ir_context: Context,
    ) -> RawJITFunc:
        """Lower ``mlir_module`` and JIT-compile ``symbol``."""
        ...


class JITContext:
    """Combine a frontend, call-boundary type mappings, and a JIT backend."""

    pyast_ctx: PyASTContext
    """Frontend used to parse Python functions into IR."""

    py_type_context: PyTypeContext
    """Python type mappings for native calls."""

    jit_backend: JITBackend
    """Backend that lowers IR and produces a :class:`~xdsl.jit.function.RawJITFunc`."""

    def __init__(self, jit_backend: JITBackend):
        """Create empty frontend and type-converter state around ``jit_backend``."""
        self.pyast_ctx = PyASTContext()
        self.py_type_context = PyTypeContext()
        self.jit_backend = jit_backend

    def jit(
        self, signature: TypeForm[Callable[P, R]]
    ) -> Callable[[Callable[P, R]], WrappedJITFunc[P, R]]:
        """
        Return a decorator that JIT-compiles a function with ``signature``.

        ``signature`` selects the Python-to-C type maps and is checked against the
        C signature derived by the backend. It is passed explicitly so annotations
        need not be evaluated.
        """

        def inner(func: Callable[P, R]) -> WrappedJITFunc[P, R]:
            parsed_program = self.pyast_ctx.parse_program(func)
            raw = self.jit_backend.jit(
                parsed_program.module,
                parsed_program.name,
                self.pyast_ctx.ir_context,
            )
            return wrap_jit_func(raw, func, signature, self.py_type_context)

        return inner


def register_builtin_type_maps(ctx: JITContext) -> None:
    """
    Register the Python ``float`` / IR ``f64`` / C ``double`` mapping.

    Updates the frontend type map and the
    :class:`~xdsl.jit.py_type_context.PyTypeContext` together. The backend registers
    the IR side on its own :class:`~xdsl.jit.c_type_context.CTypeContext`.
    """
    ctx.pyast_ctx.register_type(float, builtin.f64)
    ctx.py_type_context.register_type_map(TypeMap(float, "double"))
