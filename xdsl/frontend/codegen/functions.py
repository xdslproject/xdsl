import ast

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from xdsl.frontend.codegen.type_conversion import TypeHintConverter
from xdsl.frontend.codegen.utils.codegen_function import check_function_signature, get_argument_types, get_return_types

from xdsl.ir import Attribute

@dataclass
class FunctionVisitor(ast.NodeVisitor):

    globals: Dict[str, Any]

    hint_converter: TypeHintConverter = field(init=False)

    functions: Dict[str, Tuple[List[Attribute], List[Attribute]]] = field(init=False)

    def __init__(self, globals):
        self.functions = dict()
        self.hint_converter = TypeHintConverter(globals)

    def visit(self, node: ast.AST):
        return super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        function_name = node.name

        # TODO: we can drop check in codegen visitor with this here.
        check_function_signature(node)

        arg_types = get_argument_types(node, self.hint_converter)
        return_types = get_return_types(node, self.hint_converter)
        self.functions[function_name] = (arg_types, return_types)
