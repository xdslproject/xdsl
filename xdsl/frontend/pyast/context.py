import ast
import functools
import textwrap
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from inspect import currentframe, getsource
from sys import _getframe  # pyright: ignore[reportPrivateUsage]
from types import FrameType
from typing import Any, NamedTuple

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.program import FrontendProgram, P, PyASTProgram, R
from xdsl.frontend.pyast.utils.builder import PyASTBuilder
from xdsl.frontend.pyast.utils.python_code_check import PythonCodeCheck
from xdsl.frontend.pyast.utils.type_conversion import FunctionRegistry, TypeRegistry
from xdsl.ir import Dialect, Operation, TypeAttribute
from xdsl.passes import ModulePass, PassPipeline
from xdsl.transforms.desymref import FrontendDesymrefyPass


class FuncInfo(NamedTuple):
    """Information about a decorated function being generated into IR."""

    file: str
    """The path of the file containing the function."""

    globals: dict[str, Any]
    """The globals defined in that file up to the point of function definition."""

    ast: ast.FunctionDef
    """The Python AST representation of the function."""


def default_pipeline_callback(
    _previous_pass: ModulePass, module: ModuleOp, _next_pass: ModulePass
) -> None:
    """Default callback to verify the module after each transformation pass."""
    module.verify()


@dataclass
class PyASTContext:
    """Encapsulate the mapping between Python and IR types and operations."""

    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    """Mappings between source code and IR type."""

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    """Mappings between functions and their operation types."""

    post_transforms: list[ModulePass] = field(
        default_factory=lambda: [FrontendDesymrefyPass()]
    )
    """An ordered list of passes to apply to the built module."""

    post_callback: Callable[[ModulePass, ModuleOp, ModulePass], None] | None = (
        default_pipeline_callback
    )
    """Callback to run between post transforms."""

    ir_context: Context = field(
        default_factory=lambda: Context(allow_unregistered=True)
    )
    """The xDSL context to use when applying transformations to the built module."""

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

    def register_post_transform(self, transform: ModulePass) -> None:
        """Add a module pass to be run on the generated IR."""
        self.post_transforms.append(transform)

    def register_dialect(self, dialect: Dialect) -> None:
        """Add a dialect to the context used for transformation."""
        self.ir_context.load_dialect(dialect)

    @property
    def pass_pipeline(self) -> PassPipeline:
        """Get a pass pipeline from the context state."""
        return PassPipeline(tuple(self.post_transforms), self.post_callback)

    @classmethod
    def _get_func_info(
        cls,
        current_frame: FrameType | None,
        func: Callable[P, R],
    ) -> FuncInfo:
        """Get information about the decorated function."""
        # Get the correct function frame from the call stack
        assert current_frame is not None
        func_frame = current_frame.f_back
        assert func_frame is not None

        # Get the required information about the function from the frame
        func_file = func_frame.f_code.co_filename
        func_globals = func_frame.f_globals

        # Remove leading indentation from the source code to avoid parsing errors
        source = getsource(func.__code__)
        source = textwrap.dedent(source)

        # Retrieve the AST for the function body, without the decorator
        func_ast = ast.parse(source).body[0]
        assert isinstance(func_ast, ast.FunctionDef)
        assert func_ast.name == func.__name__
        assert len(func_ast.decorator_list) == 1
        func_ast.decorator_list = []

        # Return the information about the function
        return FuncInfo(func_file, func_globals, func_ast)

    @classmethod
    def _get_wrapped_program(
        cls, func: Callable[P, R], builder: PyASTBuilder
    ) -> PyASTProgram[P, R]:
        """Return a PyAST program for this function with the builder."""
        program = PyASTProgram[P, R](
            name=func.__name__,
            func=func,
            _builder=builder,
        )
        functools.update_wrapper(program, func)
        assert program.__doc__ == func.__doc__
        return program

    def parse_program(self, func: Callable[P, R]) -> PyASTProgram[P, R]:
        """Get a program wrapper by decorating a function."""
        func_file, func_globals, func_ast = self._get_func_info(currentframe(), func)
        builder = PyASTBuilder(
            type_registry=self.type_registry,
            function_registry=self.function_registry,
            file=func_file,
            globals=func_globals,
            function_ast=func_ast,
            build_context=self.ir_context,
            post_transforms=self.pass_pipeline,
        )
        return self._get_wrapped_program(func, builder)


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
