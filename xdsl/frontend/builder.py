import ast
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.code_generation import CodeGeneration
from xdsl.frontend.pyast.utils.python_code_check import PythonCodeCheck
from xdsl.frontend.pyast.utils.type_conversion import (
    FunctionRegistry,
    TypeConverter,
    TypeRegistry,
)
from xdsl.transforms.desymref import Desymrefier


@dataclass
class PyASTBuilder:
    """Build an operation ."""

    type_registry: TypeRegistry
    function_registry: FunctionRegistry

    file: str
    globals: dict[str, Any]
    ast: ast.FunctionDef

    desymref: bool

    def build(self) -> ModuleOp:
        # Get the functions and blocks from the builder state
        functions_and_blocks = PythonCodeCheck.run([self.ast], self.file)

        # Convert the Python AST into xDSL IR objects
        type_converter = TypeConverter(
            globals=self.globals,
            type_registry=self.type_registry,
            function_registry=self.function_registry,
        )
        module = CodeGeneration.run_with_type_converter(
            type_converter,
            functions_and_blocks,
            self.file,
        )
        module.verify()

        # Optionally run desymrefication pass to produce actual SSA
        if self.desymref:
            Desymrefier().desymrefy(module)
            module.verify()

        return module

    # @property
    # def module(self) -> ModuleOp:
    #     return ModuleOp([])
