import ast

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from xdsl.dialects.builtin import FunctionType, TensorType, UnrankedTensorType
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.codegen.type_conversion import TypeConverter
from xdsl.ir import Attribute


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

    side_effects: Dict[str, List[Tuple[int, Attribute]]] = field(default_factory=dict)
    """
    A dictionary which stores a list of function output types which have
    side-effects, for each function.
    """

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

    def _get_argument_types(self, node: ast.FunctionDef) -> List[Tuple[str, Attribute]]:
        """Converts function argument types to xDSL types."""
        arg_xdsl_types_with_name = [(arg.arg, self.converter.convert_type_hint(arg.annotation)) for arg in node.args.args]
        return arg_xdsl_types_with_name

    def _get_return_types(self, node: ast.FunctionDef) -> List[Attribute]:
        """Converts function return types to xDSL types."""
        xdsl_type = self.converter.convert_type_hint(node.returns)
        return_xdsl_types = [xdsl_type] if xdsl_type is not None else []
        return return_xdsl_types

    def _has_side_effects(self, xdsl_type: Attribute) -> bool:
        # TODO: This should be a trait rather than an isinstance check.
        if isinstance(xdsl_type, TensorType) or isinstance(xdsl_type, UnrankedTensorType):
            # TODO: Also MemRef!
            return True
        return False

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

        # In Python, we can pass values by reference, while MLIR uses SSA. In order to avoid
        # cases like:
        #
        # def foo(l: List[int]):
        #   l[0] = 2
        #
        # where list `l` remains unmodified, we opt for ensuring the function always returns
        # arguments which are passed by reference and can have side-effects.
        # TODO: We only need to track arguments which **have** side-effects, but for now MLIR
        # can take are of that and prune unused types? So having a simple if-else check should
        # be enough for now.
        arg_xdsl_types = self._get_argument_types(node)
        return_xdsl_types = self._get_return_types(node)
        self.side_effects[node.name] = []

        # TODO: what if the function return its side-effect type already? We can take care of this
        # on MLIR side I guess. Here it is not possible because value can "escape" through symref.

        idx = len(return_xdsl_types)
        for arg_name, xdsl_type in arg_xdsl_types:
            if self._has_side_effects(xdsl_type):
                # Make sure we return the types.
                if len(return_xdsl_types) == 0:
                    # Case 1: There are no return types so we have to create a new `ast.Return`.
                    new_node = ast.Return(ast.Name(arg_name))
                    node.body.append(new_node)
                else:
                    # case 2: There are some return types already. Modify the return statement.
                    # TODO: Should this be an error instead?
                    last_stmt = node.body[-1]
                    assert isinstance(last_stmt, ast.Return)
                    
                    old_return_value = last_stmt.value
                    if isinstance(old_return_value, ast.Tuple):
                        old_return_value.elts.append(ast.Name(arg_name))
                    else:
                        last_stmt.value = ast.Tuple([old_return_value, ast.Name(arg_name)])
                
                # Make sure we change the function signature.
                return_xdsl_types.append(xdsl_type)
                self.side_effects[node.name].append((idx, xdsl_type))
                idx += 1
        
        arg_xdsl_types = [xdsl_type for _, xdsl_type in arg_xdsl_types]
        self.functions[node.name] = FunctionType.from_lists(arg_xdsl_types, return_xdsl_types)
