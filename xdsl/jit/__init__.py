"""
JIT compilation of Python functions via xDSL IR.

Frontend builds mid-level IR. Backend lowers to its dialect and binds ctypes
from IR types. Context only parses, delegates, and wraps Python values.

Library authors configure a :class:`~xdsl.jit.context.JITContext` with a frontend
(:class:`~xdsl.frontend.pyast.context.PyASTContext`), ctypes bridges, and a
:class:`~xdsl.jit.context.JITBackend`. End users apply
:meth:`~xdsl.jit.context.JITContext.jit` without knowing about compilers::

    @ctx.jit(Callable[[float, float], float])
    def plus(a: float, b: float) -> float:
        return a + b

Pipeline:

1. Parse the Python function to a ``ModuleOp`` (PyAST), applying frontend
   post-transforms.
2. Hand the module to the backend, which lowers to its dialect and produces a
   :class:`~xdsl.jit.function.RawJITFunc`.
3. Wrap it as a :class:`~xdsl.jit.function.WrappedJITFunc` that marshals Python
   values through ctypes.

The ``Callable[...]`` argument to :meth:`~xdsl.jit.context.JITContext.jit` is the
ABI signature used for marshalling. It is passed explicitly so annotations need
not be evaluated.
"""
