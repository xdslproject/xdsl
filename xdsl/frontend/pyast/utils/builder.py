import ast
from dataclasses import dataclass
from typing import Any, NamedTuple

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.code_generation import CodeGeneration
from xdsl.frontend.pyast.utils.type_conversion import (
    FunctionRegistry,
    TypeConverter,
    TypeRegistry,
)
from xdsl.passes import ModulePass


class PostTransform(NamedTuple):
    """Transformation to apply to built IR."""

    module_pass: ModulePass
    """The module pass to apply."""

    verify: bool = True
    """Whether to verify the result of the pass."""


@dataclass
class PyASTBuilder:
    """Builder for xDSL modules from aspects of a Python function."""

    type_registry: TypeRegistry
    """Mappings between source code and IR type."""

    function_registry: FunctionRegistry
    """Mappings between functions and their operation types."""

    file: str | None
    """The file path of the function being built."""

    globals: dict[str, Any]
    """Global information for the function being built, including all the imports."""

    function_ast: ast.FunctionDef
    """The AST tree for the function being built."""

    build_context: Context
    """The xDSL context to use when applying transformations to the built module."""

    post_transforms: list[PostTransform]
    """An ordered list of passes to apply to the built module."""

    def build(self) -> ModuleOp:
        """Build a module from the builder state."""
        # Convert the Python AST into xDSL IR objects
        type_converter = TypeConverter(
            self.globals,
            self.type_registry,
            self.function_registry,
        )
        module = CodeGeneration.run_with_type_converter(
            type_converter,
            self.function_ast,
            self.file,
        )
        module.verify()

        context = self.build_context.clone()
        for transform, verify in self.post_transforms:
            transform.apply(context, module)
            if verify:
                module.verify()

        return module
