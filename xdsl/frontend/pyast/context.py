import ast
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from inspect import getsource
from sys import _getframe  # pyright: ignore[reportPrivateUsage]
from typing import Any, overload

from xdsl.frontend.builder import PyASTBuilder
from xdsl.frontend.pyast.program import FrontendProgram, P, PyASTProgram, R
from xdsl.frontend.pyast.utils.python_code_check import PythonCodeCheck
from xdsl.frontend.pyast.utils.type_conversion import FunctionRegistry, TypeRegistry
from xdsl.ir import Operation, TypeAttribute


@dataclass
class PyASTContext:
    """Encapsulate the mapping between Python and IR types and operations."""

    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    """Mappings between source code and IR type."""

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    """Mappings between functions and their operation types."""

    def register_type(
        self,
        source_type: type,
        ir_type: TypeAttribute,
    ) -> None:
        """Associate a type in the source code with its type in the IR."""
        self.type_registry.insert(source_type, ir_type)

    def register_function(
        self, function: Callable[..., Any], ir_constructor: Callable[..., Operation]
    ) -> None:
        """Associate a method on an object in the source code with its IR implementation."""
        self.function_registry.insert(function, ir_constructor)

    @overload
    def parse_program(
        self,
        func: None = None,
        *,
        desymref: bool = True,
    ) -> Callable[[Callable[P, R]], PyASTProgram[P, R]]: ...

    @overload
    def parse_program(
        self,
        func: Callable[P, R],
        *,
        desymref: bool = True,
    ) -> PyASTProgram[P, R]: ...

    def parse_program(
        self,
        func: Callable[P, R] | None = None,
        *,
        desymref: bool = True,
    ) -> Callable[[Callable[P, R]], PyASTProgram[P, R]] | PyASTProgram[P, R]:
        """Get a program wrapper by decorating a function."""

        def decorator(func: Callable[P, R]) -> PyASTProgram[P, R]:
            builder = PyASTBuilder(
                type_registry=self.type_registry,
                function_registry=self.function_registry,
                desymref=desymref,
            )
            return PyASTProgram[P, R](
                name=func.__name__,
                func=func,
                _builder=builder,
            )

        if func is None:
            return decorator
        return decorator(func)


@dataclass
class CodeContext(AbstractContextManager[Any]):
    """
    The CodeContext with block marks the scope in which the code in the custom
    DSL can be written. This code will be translated to xDSL/MLIR.
    """

    program: FrontendProgram
    """
    Underlying frontend program which can be compiled and translated to
    xDSL/MLIR.
    """

    def __enter__(self) -> None:
        # First, get the Python AST from the code.
        frame = _getframe(1)
        self.program.file = frame.f_code.co_filename
        src = getsource(frame)
        python_ast = ast.parse(src)

        # Get all the global information and record it as well. In particular,
        # this contains all the imports.
        self.program.globals = frame.f_globals

        # Find where the program starts.
        for node in ast.walk(python_ast):
            if (
                isinstance(node, ast.With)
                and node.lineno == frame.f_lineno - frame.f_code.co_firstlineno + 1
            ):
                # Found the program AST. Store it for later compilation or
                # execution.
                self.program.stmts = node.body

    def __exit__(self, *args: object):
        # Having proccessed all the code in the context, check it is well-formed
        # and can be compiled/executed. Additionally, record it for subsequent code generation.
        assert self.program.stmts is not None
        self.program.functions_and_blocks = PythonCodeCheck.run(
            self.program.stmts, self.program.file
        )
