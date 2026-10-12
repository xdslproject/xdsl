import ast
import functools
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import getsourcelines
from typing import Any, NamedTuple

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.program import P, PyASTProgram, R
from xdsl.frontend.pyast.utils.builder import PyASTBuilder
from xdsl.frontend.pyast.utils.type_conversion import (
    FunctionRegistry,
    LiteralRegistry,
    TypeRegistry,
)
from xdsl.ir import Dialect, Operation, TypeAttribute
from xdsl.passes import ModulePass, PassPipeline, PassPipelineCallbackType
from xdsl.transforms.desymref import FrontendDesymrefyPass
from xdsl.utils.lexer import Location


class FuncInfo(NamedTuple):
    """Information about a decorated function being generated into IR."""

    source_location: Location
    """The original file location of the start of the dedented source."""

    globals: dict[str, Any]
    """The globals defined in that file up to the point of function definition."""

    ast: ast.FunctionDef
    """The Python AST representation of the function."""


def default_pipeline_callback(
    previous_pass: ModulePass | None, module: ModuleOp, next_pass: ModulePass | None
) -> None:
    """Default callback to verify the module after each transformation pass."""
    if previous_pass and next_pass:
        module.verify()


@dataclass
class PyASTContext:
    """Encapsulate the mapping between Python and IR types and operations."""

    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    """Mappings between source code and IR type."""

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    """Mappings between functions and their operation types."""

    literal_registry: LiteralRegistry = field(default_factory=LiteralRegistry)
    """Mappings between literal types and their operation constructors."""

    post_transforms: list[ModulePass] = field(
        default_factory=lambda: [FrontendDesymrefyPass()]
    )
    """An ordered list of passes to apply to the built module."""

    post_callback: PassPipelineCallbackType | None = default_pipeline_callback
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

    def register_literal(
        self, value_type: type[object], ir_constructor: Callable[[Any], Operation]
    ) -> None:
        """Associate a Python literal type with an IR constructor."""
        self.literal_registry.insert(value_type, ir_constructor)

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
    def _get_func_info(cls, func: Callable[P, R]) -> FuncInfo:
        """Get information about the decorated function."""
        func_file = func.__code__.co_filename
        func_globals = func.__globals__

        # Remove leading indentation from the source code to avoid parsing errors
        source_lines, start_line = getsourcelines(func.__code__)
        source = "".join(source_lines)
        dedented_source = textwrap.dedent(source)
        # Dedent removes the same ASCII whitespace prefix from non-blank lines.
        indentation = len(source.partition("\n")[0]) - len(
            dedented_source.partition("\n")[0]
        )
        source_location = Location(func_file, start_line, indentation + 1)

        # Retrieve the AST for the function body, without the decorator
        func_ast = ast.parse(dedented_source).body[0]
        assert isinstance(func_ast, ast.FunctionDef)
        assert func_ast.name == func.__name__
        assert len(func_ast.decorator_list) == 1
        func_ast.decorator_list = []

        # Return the information about the function
        return FuncInfo(source_location, func_globals, func_ast)

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
        source_location, func_globals, func_ast = self._get_func_info(func)
        builder = PyASTBuilder(
            type_registry=self.type_registry,
            function_registry=self.function_registry,
            literal_registry=self.literal_registry,
            source_location=source_location,
            globals=func_globals,
            function_ast=func_ast,
            build_context=self.ir_context,
            post_transforms=self.pass_pipeline,
        )
        return self._get_wrapped_program(func, builder)
