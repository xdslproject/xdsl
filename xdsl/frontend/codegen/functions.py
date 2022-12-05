import ast

from dataclasses import dataclass, field
from typing import Dict
from xdsl.dialects.builtin import FunctionType
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.codegen.type_conversion import TypeConverter


@dataclass
class FunctionVisitor(ast.NodeVisitor):
    """
    This class is responsible for finding all functions defined in this code
    context block. Should be called at 'builtin.module' level.
    """

    converter: TypeConverter
    """Converts source Python/front-end types to xDSL."""

    functions: Dict[str, FunctionType] = field(default_factory=dict)
    """A dictionary of found functions and associated type signatures."""

    def _is_function_valid(self, node: ast.FunctionDef):
        """Throws an exception if this function cannot be lowered to xDSL."""

        # Don't support vararg and its friends.
        if getattr(node.args, "vararg") is not None:
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has 'vararg' but is not supposed to.")
        if getattr(node.args, "kwarg") is not None:
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has 'kwarg' but is not supposed to.")
        if getattr(node.args, "kwonlyargs"):
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has 'kwonlyargs' but is not supposed to.")
        if getattr(node.args, "kw_defaults"):
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has 'kw_defaults' but is not supposed to.")
        if getattr(node.args, "defaults"):
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has 'defaults' but is not supposed to.")

        # Explicitly require type annotations on all function arguments.
        not_annotated = []
        for i, arg in enumerate(node.args.args):
            annotation = arg.annotation
            arg.col_offset
            if annotation is None:
                not_annotated.append(i)

        # TODO: Note that we do not require type annotation for return type, simply because
        # writing `foo() -> None` does not seem that great. Maybe we should?
        
        # Check did not pass, raise an error.
        if len(not_annotated) > 0:
            p = "position " if len(not_annotated) == 1 else "positions "
            positions = ",".join(not_annotated)
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has non-annotated arguments at {p}{positions}.")

    def _get_argument_types(self, node: ast.FunctionDef):
        """Converts function argument types to xDSL types."""
        arg_xdsl_types = [self.converter.convert_type_hint(arg.annotation) for arg in node.args.args]
        return arg_xdsl_types

    def _get_return_types(self, node: ast.FunctionDef):
        """Converts function return types to xDSL types."""
        xdsl_type = self.converter.convert_type_hint(node.returns)
        return_xdsl_types = [xdsl_type] if xdsl_type is not None else []
        return return_xdsl_types

    def visit(self, node: ast.AST):
        return super().visit(node)

    def visit_With(self, node: ast.With):
        # TODO: This assumes that there are no regions, so with block defines a module.
        # In the future, this can change, so make sure to support this here and in code
        # generation visitor.
        # For example, we may want to consider having a nested dictionary to imitate the
        # scoping of function calls, but strictly speaking this can be done at call time.
        return

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Let's check if the function signature is valid straight away. Then, during
        # code generation is suffices to reuse function types.
        self._is_function_valid(node)

        arg_xdsl_types = self._get_argument_types(node)
        return_xdsl_types = self._get_return_types(node)
        self.functions[node.name] = FunctionType.from_lists(arg_xdsl_types, return_xdsl_types)
