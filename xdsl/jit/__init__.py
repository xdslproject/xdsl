"""
JIT compilation of Python functions via xDSL IR.

The frontend builds IR. The backend lowers and compiles it, derives a C function
signature from the IR types, and binds the native entry point. The context
coordinates parsing, compilation, and call wrapping.

A :class:`~xdsl.jit.context.JITContext` combines a frontend
(:class:`~xdsl.frontend.pyast.context.PyASTContext`), Python-to-C type mappings,
and a :class:`~xdsl.jit.context.JITBackend`. End users apply
:meth:`~xdsl.jit.context.JITContext.jit` without knowing about compilers::

    @ctx.jit(Callable[[float, float], float])
    def plus(a: float, b: float) -> float:
        return a + b

Pipeline:

1. Parse the Python function to a ``ModuleOp`` (PyAST), applying frontend
   post-transforms.
2. Hand the module to the backend, which lowers to its dialect and produces a
   :class:`~xdsl.jit.function.RawJITFunc`.
3. Wrap it as a :class:`~xdsl.jit.function.WrappedJITFunc` that converts arguments
   and results according to the registered type mappings and invokes the native
   entry point.

The ``Callable[...]`` argument to :meth:`~xdsl.jit.context.JITContext.jit`
describes the Python call signature. It selects the registered value conversions
and must match the C signature derived by the backend from the IR function type.
"""
