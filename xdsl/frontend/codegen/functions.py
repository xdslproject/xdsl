from __future__ import annotations

import ast
import copy

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from xdsl.dialects.builtin import FunctionType, TensorType, UnrankedTensorType
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.codegen.type_conversion import TypeConverter
from xdsl.ir import Attribute


@dataclass
class ArgInfo:
    """Stores all information about the argument values."""

    poisition: int
    name: str
    xdsl_type: Attribute
    has_side_effects: bool
    template: bool


@dataclass
class ReturnInfo:
    """Stores all information about the return values."""

    poisition: int
    xdsl_type: Attribute


@dataclass
class FunctionInfo:
    """Stores information about the function."""

    ast_node: ast.FunctionDef
    """Pointer to AST of this function."""

    arg_info: List[ArgInfo]
    """Argument information."""

    return_info: List[ReturnInfo]
    """Return values information."""

    has_side_effects: bool
    """True if this function has side-effect arguments."""

    template: bool
    """True if this function is a template."""

    template_instantiation: bool
    """True if this function is a template instantiation."""
    
    @staticmethod
    def new(ast_node) -> FunctionInfo:
        return FunctionInfo(ast_node, [], [], False, False, False)

@dataclass
class LocalFunctionAnalyzer(ast.NodeVisitor):
    """
    This class analyzes local user-defined functions, e.g. which arguments are side effect, or
    whether the function is a template.
    """

    converter: TypeConverter
    """Converts source Python/front-end types to xDSL."""

    function_infos: Dict[str, FunctionInfo] = field(default_factory=dict)

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
        function_name = node.name
        function_info = FunctionInfo.new(node)
        self.function_infos[function_name] = function_info

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
        not_annotated_positions = []
        for i, arg in enumerate(node.args.args):
            annotation = arg.annotation
            arg.col_offset
            if annotation is None:
                not_annotated_positions.append(i)

        # TODO: Note that we do not require type annotation for return type, simply because writing `foo() -> None` does not seem
        # that great. Maybe we should?

        # Check did not pass, raise an error.
        if len(not_annotated_positions) > 0:
            p = "position " if len(not_annotated_positions) == 1 else "positions "
            positions = ",".join(not_annotated_positions)
            raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has non-annotated arguments at {p}{positions}.")

        # Function can also be a template. Find which arguments are templated. 
        template_arguments: Set[str] = None
        num_decorators = len(node.decorator_list)
        if num_decorators != 0:
            # All templates must have a well-defined decorator.
            if num_decorators != 1:
                raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has {num_decorators} but can only have 1.")
            if not isinstance(node.decorator_list[0], ast.Call) or not isinstance(node.decorator_list[0].func, ast.Name) or node.decorator_list[0].func.id != "template":
                raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has unknown decorator. For decorating the function as a template, use '@template(..)'.")
            
            template_arguments = set()
            wrong_template_argument_positions: List[int] = []
            for i, arg in enumerate(node.decorator_list[0].args):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    template_arguments.add(arg.value)
                else:
                    wrong_template_argument_positions.append(i)
            
            # All template arguments are defined as a list of strings passed to the decorator.
            if len(wrong_template_argument_positions) > 0:
                p = "position " if len(wrong_template_argument_positions) == 1 else "positions "
                positions = ",".join(wrong_template_argument_positions)
                raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has unknown template arguments at {p}{positions}. All template arguments should be string constnats, passed to the decorator, e.g. '@template(\"A\", \"B\", \"C\")'.")

        num_template_args = 0
        num_side_effect_args = 0
        for i, arg in enumerate(node.args.args):
            # TODO: Use traits for side-effect types, and also add other types like MemRef here.
            xdsl_type = self.converter.convert_type_hint(arg.annotation)
            has_side_effects = isinstance(xdsl_type, TensorType) or isinstance(xdsl_type, UnrankedTensorType)

            name = arg.arg
            if template_arguments is not None and name in template_arguments:
                # Templated arguments cannot have side effects.
                num_template_args += 1
                function_info.arg_info.append(ArgInfo(i, name, xdsl_type, False, True))
                continue

            function_info.arg_info.append(ArgInfo(i, name, xdsl_type, has_side_effects, False))
            if has_side_effects:
                num_side_effect_args += 1
        
        function_info.has_side_effects = num_side_effect_args > 0
        function_info.template = num_template_args > 0

        # TODO: this assumes that hint conversion always returns a concrete xDSL type. However, if we
        # support tuples, this may not hold anymore, so we have to be carful here.
        xdsl_type = self.converter.convert_type_hint(node.returns)
        if xdsl_type is not None:
            function_info.return_info.append(ReturnInfo(0, xdsl_type))


@dataclass
class LocalCallAnalyzer(ast.NodeVisitor):

    function_infos: Dict[str, FunctionInfo]

    def visit(self, node: ast.AST):
        return super().visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = node.func.id

        # We only care about local user-defined functions.
        if func_name not in self.function_infos:
            return
        function_info = self.function_infos[func_name]

        # We do actual type checking during code generation, however here we can at least
        # catch cases where we have a wrong number of arguments.
        if len(node.args) != len(function_info.arg_info):
            raise CodegenException(node.lineno, node.col_offset, f"Call to function {func_name} has {len(node.args)} arguments, but expects {len(function_info.arg_info)}.")

        # If this is a call to a template function, we have to make sure we will instantiate the template
        # before the code generation. 
        if function_info.template:
            
            parameters: Dict[str, ast.expr] = dict()
            mangled_func_name = func_name
            non_template_args = []
            non_template_arg_names = []
            non_template_arg_indices = []

            for i, (arg, arg_info) in enumerate(zip(node.args, function_info.arg_info)):
                if arg_info.template:
                    # Try to instantiate the template argument, and raise an exeception if something goes wrong.
                    try:
                        value = eval(ast.unparse(arg))
                    except TypeError as e:
                        raise CodegenException(arg.lineno, arg.col_offset, f"Invalid template instantiation, TypeError: {e}")

                    parameters[arg_info.name] = ast.Constant(value)
                    mangled_func_name += f"__{value}"
                else:
                    non_template_arg_names.append(arg_info.name)
                    non_template_arg_indices.append(i)
                    non_template_args.append(arg)
            
            # Change the name of the function and use only its non-template arguments.
            # node.args = non_template_args
            # node.func = ast.Name(mangled_func_name)

            # If this template has not been instantiated before, do so. This can be easily done by copying
            # the old function node, adapting its arguments and replacing 
            if mangled_func_name not in self.function_infos:
                template_function_info = self.function_infos[func_name]
                template_node = copy.deepcopy(template_function_info.ast_node)
                
                # Modify AST first.
                template_node.decorator_list = []
                template_node.name = mangled_func_name

                args = []
                for i, arg in enumerate(template_node.args.args):
                    if i in non_template_arg_indices:
                        args.append(arg)
                template_node.args.args = args

                for stmt in template_node.body:
                    visitor = ReplaceVisitor(parameters)
                    visitor.visit(stmt)

                # Make sure to change the function info.
                new_function_info = FunctionInfo.new(template_node)

                idx = 0
                new_arg_info: List[ArgInfo] = []
                for arg_info in template_function_info.arg_info:
                    if not arg_info.template:
                        new_arg_info.append(ArgInfo(idx, arg_info.name, arg_info.xdsl_type, arg_info.has_side_effects, False))
                        idx += 1

                new_function_info.arg_info = new_arg_info
                new_function_info.return_info = template_function_info.return_info
                new_function_info.has_side_effects = template_function_info.has_side_effects
                new_function_info.template = False
                new_function_info.template_instantiation = True

                self.function_infos[mangled_func_name] = new_function_info


@dataclass
class ReplaceVisitor(ast.NodeVisitor):

    parameters: Dict[str, ast.Constant]

    def check_node(self, node: ast.AST) -> bool:
        if isinstance(node, ast.AnnAssign):
            target = node.target
            while isinstance(target, ast.Subscript):
                target = target.value
            
            if isinstance(target, ast.Name):
                if target.id in self.parameters:
                    raise CodegenException(node.lineno, node.col_offset, f"Cannot assign to template parameter '{target.id}'.")

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in self.parameters:
                        raise CodegenException(node.lineno, node.col_offset, f"Cannot assign to template parameter '{target.id}'.")

    def visit(self, node: ast.AST):
        self.check_node(node)
        for field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                if isinstance(field_value, ast.Name) and field_value.id in self.parameters:
                    setattr(node, field_name, self.parameters[field_value.id])
                self.visit(field_value)
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if isinstance(item, ast.AST):
                        if isinstance(item, ast.Name) and item.id in self.parameters:
                            field_value[i] = self.parameters[item.id]
                        self.visit(item)
