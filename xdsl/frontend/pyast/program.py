import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Final, Generic, ParamSpec

from typing_extensions import TypeVar

from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.code_generation import CodeGeneration
from xdsl.frontend.pyast.utils.builder import PyASTBuilder
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException
from xdsl.frontend.pyast.utils.python_code_check import FunctionMap
from xdsl.frontend.pyast.utils.type_conversion import (
    FunctionRegistry,
    TypeConverter,
    TypeRegistry,
)
from xdsl.ir import Operation, TypeAttribute
from xdsl.printer import Printer
from xdsl.transforms.desymref import Desymrefier

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class PyASTProgram(Generic[P, R]):
    """Wrapper to associate an IR representation with a Python function."""

    name: Final[str]
    """The name of the function describing the program."""

    func: Final[Callable[P, R]]
    """A callable object for the function describing the program."""

    _builder: Final[PyASTBuilder]
    """An internal object to contextually build an IR module from the function."""

    _module: ModuleOp | None = None
    """An internal object to cache the built IR module."""

    @property
    def module(self) -> ModuleOp:
        """Lazily build the module when required, once."""
        if self._module is None:
            self._module = self._builder.build()
        return self._module

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Pass through calling the object to its Python implementation."""
        return self.func(*args, **kwargs)


@dataclass
class FrontendProgram:
    """
    Class to store the Python AST of a program written in the frontend. This
    program can be compiled and translated to xDSL or MLIR.
    """

    stmts: list[ast.stmt] | None = field(default=None)
    """Input AST nodes."""

    functions_and_blocks: FunctionMap | None = field(default=None)
    """Processed AST nodes stored for code generation."""

    globals: dict[str, Any] | None = field(default=None)
    """Global information for this program, including all the imports."""

    xdsl_program: ModuleOp | None = field(default=None)
    """Generated xDSL program when AST is compiled."""

    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    """Mappings between source code and IR type."""

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    """Mappings between functions and their operation types."""

    file: str | None = field(default=None)
    """Path to the file that contains the program."""

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

    def _check_can_compile(self):
        if self.stmts is None or self.globals is None:
            msg = """
Cannot compile program without the code context. Try to use:
    p = FrontendProgram()
    with CodeContext(p):
        # Your code here."""
            raise FrontendProgramException(msg)

    def compile(self, desymref: bool = True) -> None:
        """Generates xDSL from the source program."""

        # Both statements and globals msut be initialized from within the
        # `CodeContext`.
        self._check_can_compile()
        assert self.globals is not None
        assert self.functions_and_blocks is not None

        type_converter = TypeConverter(
            globals=self.globals,
            type_registry=self.type_registry,
            function_registry=self.function_registry,
        )
        self.xdsl_program = CodeGeneration.run_with_type_converter(
            type_converter,
            self.functions_and_blocks,
            self.file,
        )
        self.xdsl_program.verify()

        # Optionally run desymrefication pass to produce actual SSA.
        if desymref:
            self.desymref()

    def desymref(self) -> None:
        """Desymrefy the generated xDSL."""
        assert self.xdsl_program is not None
        Desymrefier().desymrefy(self.xdsl_program)
        self.xdsl_program.verify()

    def _check_can_print(self):
        if self.xdsl_program is None:
            msg = """
Cannot print the program IR without compiling it first. Make sure to use:
    p = FrontendProgram()
    with CodeContext(p):
        # Your code here.
    p.compile()"""
            raise FrontendProgramException(msg)

    def textual_format(self) -> str:
        self._check_can_print()
        assert self.xdsl_program is not None

        file = StringIO("")
        printer = Printer(stream=file)
        printer.print_op(self.xdsl_program)
        return file.getvalue().strip()
