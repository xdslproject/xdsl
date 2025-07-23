import ast
import functools
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from inspect import currentframe, getsource
from sys import _getframe  # pyright: ignore[reportPrivateUsage]
from types import FrameType
from typing import Any, overload

from xdsl.frontend.pyast.program import FrontendProgram, P, PyASTProgram, R
from xdsl.frontend.pyast.utils.builder import PyASTBuilder
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

    @classmethod
    def _get_func_info(
        cls,
        current_frame: FrameType,
        func: Callable[P, R],
        decorated_func: Callable[P, R] | None,
    ) -> tuple[str, dict[str, Any], ast.FunctionDef]:
        """Get information about the decorated function."""
        func_frame = current_frame.f_back
        if decorated_func is not None:
            assert func_frame is not None
            func_frame = func_frame.f_back
        assert func_frame is not None

        # Get the required information about the function from the frame
        func_file = func_frame.f_code.co_filename
        func_globals = func_frame.f_globals

        # Retrieve the AST for the function body, without the decorator
        func_ast = ast.parse(getsource(func.__code__)).body[0]
        assert isinstance(func_ast, ast.FunctionDef)
        assert func_ast.name == func.__name__
        assert len(func_ast.decorator_list) == 1
        func_ast.decorator_list = []

        # Return the information about the function
        return (func_file, func_globals, func_ast)

    @overload
    def parse_program(
        self,
        decorated_func: None = None,
        *,
        desymref: bool = True,
    ) -> Callable[[Callable[P, R]], PyASTProgram[P, R]]: ...

    @overload
    def parse_program(
        self,
        decorated_func: Callable[P, R],
        *,
        desymref: bool = True,
    ) -> PyASTProgram[P, R]: ...

    def parse_program(
        self,
        decorated_func: Callable[P, R] | None = None,
        *,
        desymref: bool = True,
    ) -> Callable[[Callable[P, R]], PyASTProgram[P, R]] | PyASTProgram[P, R]:
        """Get a program wrapper by decorating a function."""

        def decorator(func: Callable[P, R]) -> PyASTProgram[P, R]:
            # Get the frame as the function is being decorated
            current_frame = currentframe()
            assert current_frame is not None
            func_file, func_globals, func_ast = self._get_func_info(
                current_frame, func, decorated_func
            )

            # Construct the lazy builder with this information
            builder = PyASTBuilder(
                type_registry=self.type_registry,
                function_registry=self.function_registry,
                file=func_file,
                globals=func_globals,
                function_ast=func_ast,
                desymref=desymref,
            )

            # Return a PyAST program for this function with the builder
            program = PyASTProgram[P, R](
                name=func.__name__,
                func=func,
                _builder=builder,
            )
            functools.update_wrapper(program, func)
            assert program.__doc__ == func.__doc__
            return program

        if decorated_func is None:
            return decorator
        return decorator(decorated_func)


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
