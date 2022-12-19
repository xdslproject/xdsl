from __future__ import annotations

import ast
import copy

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple
from xdsl.dialects.builtin import FunctionType, TensorType, UnrankedTensorType
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.codegen.type_conversion import TypeConverter
from xdsl.ir import Attribute


@dataclass
class ArgInfo:
    """Stores all information about the argument values."""

    poisition: int
    """Position of the argument in the type signature of the function."""

    name: str
    """The name of the argument. Useful for naming of variables, templates and many other things."""

    xdsl_type: Attribute
    """The xDSL type of the argument."""

    has_side_effects: bool
    """
    True if the argument can have side-effects. For example, values like tensors are side-effect arguments,
    because they can be modified by the function. All primitive types like int, float, etc. have no side-effects.
    """

    template: bool
    """True if this is a template argument."""


@dataclass
class RetInfo:
    """
    Stores all information about the return values. In particular, this means that we never really have a
    tuple type, but rather the function returns multiple types instead.
    """

    poisition: int
    """Position of the return value in the type signature of the function."""

    xdsl_type: Attribute
    """The xDSL type of the return value."""


class TemplateInfo(Enum):
    NONE = 1
    TEMPLATE = 2
    TEMPLATE_INSTANTIATION = 3


@dataclass
class FunctionInfo:
    """Stores information about the function."""

    ast_node: ast.FunctionDef
    """Pointer to AST of this function."""

    arg_info: List[ArgInfo] = field(default_factory=list)
    """Argument information."""

    ret_info: List[RetInfo] = field(default_factory=list)
    """Return values information."""

    has_side_effects: bool = field(default=False)
    """True if this function has side-effect arguments."""

    template_info: TemplateInfo = field(default=TemplateInfo.NONE)
    """Stores information whether this function is a template, template instantiation, etc."""

    def is_template(self) -> bool:
        return self.template_info == TemplateInfo.TEMPLATE

    def is_template_instantiation(self) -> bool:
        return self.template_info == TemplateInfo.TEMPLATE_INSTANTIATION


@dataclass
class LocalFunctionAnalyzer:
    """
    Responsible for local function analysis, including:
      - checking which function arguments have side-effects,
      - finding templates,
      - creating new AST nodes for template instantiations.
    """

    tc: TypeConverter
    """Converts source Python/front-end types to xDSL."""

    analysis: Dict[str, FunctionInfo] = field(default_factory=dict)
    """Stores all information about visited functions."""

    @staticmethod
    def run_with_type_converter(tc: TypeConverter, stmts: List[ast.stmt]) -> Dict[str, FunctionInfo]:
        lfa = LocalFunctionAnalyzer(tc)

        # First, analyze all functions. This includes figuring out which functions have side-effects,
        # which are templates, and running type conversion on function type signatures.
        fv = FunctionVisitor(lfa.tc, lfa.analysis)
        for stmt in stmts:
            fv.visit(stmt)

        # When all the analysis for the local user-defined functions is done, we want to find out
        # which templates are instantiated. For that, we have to check every function call and instantiate
        # the template if necessary.
        cv = CallVisitor(lfa.analysis)
        for stmt in stmts:
            cv.visit(stmt)

        return lfa.analysis


@dataclass
class FunctionVisitor(ast.NodeVisitor):
    """
    This class analyzes local user-defined functions, e.g. which arguments are side-effect, or
    whether the function is a template.
    """

    converter: TypeConverter
    """Converts source Python/front-end types to xDSL."""

    analysis: Dict[str, FunctionInfo]
    """Stores all information about the visited functions."""

    def visit(self, node: ast.AST):
        return super().visit(node)

    def visit_With(self, node: ast.With):
        # TODO: We do not really support scoping, for example if we have nested modules. For now, make sure we
        # abort in these cases.
        # TODO: This breaks some FileCheck tests. THey can be re-enabled once we have proper scoping support. Anyway,
        # it is better to be too conservative then just compile wrong code!
        raise CodegenException(node.lineno, node.col_offset, f"Module-level analysis is not yet supported. Try to put all the functions in 'CodeContext' block.")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        function_name = node.name
        function_info = FunctionInfo(node)
        self.analysis[function_name] = function_info

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
        not_annotated_positions: List[int] = []
        for i, arg in enumerate(node.args.args):
            if arg.annotation is None:
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
                raise CodegenException(node.lineno, node.col_offset, f"Function '{node.name}' has {num_decorators} decorators but can only have 1 to mark it as a template.")
            if not isinstance(node.decorator_list[0], ast.Call) or not isinstance(node.decorator_list[0].func, ast.Name) or node.decorator_list[0].func.id != "template":
                raise CodegenException(node.lineno, node.col_offset, f"Function '{node.name}' has unknown decorator. For decorating the function as a template, use '@template(..)'.")
            
            if len(node.decorator_list[0].args) == 0:
                raise CodegenException(node.lineno, node.col_offset, f"Template for function '{node.name}' must have at least one template argument.")
            if len(node.decorator_list[0].args) > len(node.args.args):
                template_argument_str = "argument" if len(node.decorator_list[0].args) == 1 else "arguments"
                argument_str = "argument" if len(node.args.args) == 1 else "arguments"
                raise CodegenException(node.lineno, node.col_offset, f"Template for function '{node.name}' has {len(node.decorator_list[0].args)} template {template_argument_str}, but function expects only {len(node.args.args)} {argument_str}.")

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
                raise CodegenException(node.lineno, node.col_offset, f"Function {node.name} has unknown template arguments at {p}{positions}. All template arguments should be string constants, passed to the decorator, e.g. '@template(\"A\", \"B\", \"C\")'.")

        # Go through all arguments one by one, and find out which ones have side-effects, which ones are template arguments.
        num_template_args = 0
        num_side_effect_args = 0
        for i, arg in enumerate(node.args.args):

            # TODO: Use traits for side-effect types, and also add support types like MemRef here.
            xdsl_type = self.converter.convert_type_hint(arg.annotation)
            has_side_effects = isinstance(xdsl_type, TensorType) or isinstance(xdsl_type, UnrankedTensorType)

            name = arg.arg
            if template_arguments is not None and name in template_arguments:
                # Templated arguments cannot have side effects, as these are compile-time constants.
                num_template_args += 1
                function_info.arg_info.append(ArgInfo(i, name, xdsl_type, False, True))
                continue

            function_info.arg_info.append(ArgInfo(i, name, xdsl_type, has_side_effects, False))
            if has_side_effects:
                num_side_effect_args += 1
        
        # Make sure the naming was consistent.
        if num_decorators != 0 and len(node.decorator_list[0].args) != num_template_args:
            raise CodegenException(node.lineno, node.col_offset, f"Template for function '{node.name}' has unused template arguments. All template arguments must be named exactly the same as the corresponding function arguments.")

        function_info.has_side_effects = num_side_effect_args > 0
        function_info.template_info = TemplateInfo.TEMPLATE if num_template_args > 0 else TemplateInfo.NONE

        # TODO: this assumes that hint conversion always returns a concrete xDSL type. However, if we
        # support tuples, this may not hold anymore, so we have to be carful here.
        xdsl_type = self.converter.convert_type_hint(node.returns)
        if xdsl_type is not None:
            function_info.ret_info.append(RetInfo(0, xdsl_type))


@dataclass
class CallVisitor(ast.NodeVisitor):
    """
    This class analyzes the call sites of local functions to make sure templates are instantiated as AST nodes.
    """

    analysis: Dict[str, FunctionInfo]
    """
    Stores all information about the visited functions. Note that this analysis should be
    pre-populated by 'FunctionVisitor'.
    """

    def visit(self, node: ast.AST):
        return super().visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = node.func.id

        # We only care about local user-defined functions, which should be in analysis.
        if func_name not in self.analysis:
            return
        function_info = self.analysis[func_name]

        # We do actual type checking during code generation, however here we can at least
        # catch cases where we have a wrong number of arguments. Let's do this straight away.
        if len(node.args) != len(function_info.arg_info):
            raise CodegenException(node.lineno, node.col_offset, f"Call to function {func_name} has {len(node.args)} arguments, but expects {len(function_info.arg_info)}.")

        # If this is a call to a template function, we have to make sure we will instantiate the template
        # before the code generation. The template is a copy of the Python AST node from the source, but with
        # template arguments replaced with evaluated expressions.
        if function_info.is_template():

            # Stores evaluated paramters for this template.
            parameters: Dict[str, ast.expr] = dict()

            # The name of the template instantiation is mangled. Here, we opt for appending
            # "_PARAMETER_VALUE" for each template parameter.
            mangled_func_name = func_name

            non_template_args = []
            non_template_arg_indices = []

            for i, arg, arg_info in zip(range(len(node.args)), node.args, function_info.arg_info):
                if arg_info.template:
                    # Try to instantiate the template argument, and raise an exeception if something goes wrong.
                    try:
                        # TODO: This does not always work. Consider the case when we have:
                        #   @template("X")
                        #   def bar(X: int) -> int:
                        #       return X
                        #   @template("A", "B")
                        #   def foo(A: int, x: int, B: int) -> int:
                        #       return A - x + bar(A+B)
                        # Here, `bar(B)` is fined because `A+B` is known at compile time. But `A+x` is not. Currently both cases fail
                        # with `NameError`, but we need to 1) support the first case 2) have a more meaningful error in the second case.
                        # For (1), it is likely we need a some kind of topological sort to find out how expressions/instantiations depend
                        # on each other.
                        value = eval(ast.unparse(arg))
                    except Exception as e:
                        raise CodegenException(arg.lineno, arg.col_offset, f"Invalid template instantiation for function '{func_name}'; {type(e).__name__}: {e}")

                    parameters[arg_info.name] = ast.Constant(value)

                    # TODO: it should be relatively easy to support non-primitive type parameters, but for that we have to:
                    #   1. Think about how to mangle the name of the function (e.g. hash the list?)
                    #   2. Instead of ast.Constant create something else, like ast.List?
                    #   3. Make sure to add tests and remove tests that treat this case as invalid.
                    if not isinstance(value, int) and not isinstance(value, bool) and not isinstance(value, float):
                        raise CodegenException(node.lineno, node.col_offset, f"Call to function '{func_name}' has non-primitive template argument of type '{type(value).__name__}' at position {i}. Only primitive type arguments like int or float are supported at the moment.") 

                    parameters[arg_info.name] = ast.Constant(value)
                    mangled_func_name += f"_{value}"
                else:
                    non_template_arg_indices.append(i)
                    non_template_args.append(arg)

            # If this template has not been instantiated before, do so. This can be easily done by copying
            # the old function node, adapting its arguments and replacing the named nodes with evaluated expressions.
            if mangled_func_name not in self.analysis:
                template_function_info = self.analysis[func_name]
                template_node = copy.deepcopy(template_function_info.ast_node)
                template_node = ast.copy_location(template_node, template_function_info.ast_node)

                # Modify AST first. No decorators (i.e. it is a template instantiation) and the name is mangled.
                template_node.decorator_list = []
                template_node.name = mangled_func_name
                template_node.args.args = [template_node.args.args[idx] for idx in non_template_arg_indices]

                # Replace all template arguments with evaluated expressions.
                for stmt in template_node.body:
                    visitor = ReplaceVisitor(parameters, template_node.name)
                    visitor.visit(stmt)

                # Make sure to change the function info.
                new_function_info = FunctionInfo(template_node)

                idx = 0
                has_side_effects = False
                new_arg_info: List[ArgInfo] = []
                for arg_info in template_function_info.arg_info:
                    if not arg_info.template:
                        has_side_effects |= arg_info.has_side_effects
                        new_arg_info.append(ArgInfo(idx, arg_info.name, arg_info.xdsl_type, arg_info.has_side_effects, False))
                        idx += 1

                new_function_info.arg_info = new_arg_info
                new_function_info.ret_info = template_function_info.ret_info
                new_function_info.has_side_effects = has_side_effects
                new_function_info.template_info = TemplateInfo.TEMPLATE_INSTANTIATION

                self.analysis[mangled_func_name] = new_function_info
            
            # Lastly, replace the call with the call to template instantiation.
            node.func = ast.Name(mangled_func_name)
            node.args = non_template_args


@dataclass
class ReplaceVisitor(ast.NodeVisitor):
    """
    Helper visitor to replace all occurences of 'ast.Name' for the given template arguments
    with constant expressions provided by the template instantiation.
    """

    parameters: Dict[str, ast.Constant]
    """Template arguments and their values (evaluated expressions)."""

    current_function_name: str
    """Used to print a comprehensive error message."""

    def check_node(self, node: ast.AST) -> bool:
        """
        Checks if the node uses template arguments correctly. For example, one cannot assign to
        a template variable, etc.
        """

        if isinstance(node, ast.AnnAssign):
            target = node.target
            while isinstance(target, ast.Subscript):
                target = target.value
            
            if isinstance(target, ast.Name):
                if target.id in self.parameters:
                    raise CodegenException(node.lineno, node.col_offset, f"Cannot redefine the template parameter '{target.id}' in function '{self.current_function_name}'.")

        # No assignments like `N = ...` or `N[i][j] = ...`.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                n = target
                while isinstance(n, ast.Subscript):
                    n = n.value
                
                if isinstance(n, ast.Name):
                    if n.id in self.parameters:
                        raise CodegenException(node.lineno, node.col_offset, f"Cannot assign to template parameter '{n.id}' in function '{self.current_function_name}'.")

    def visit(self, node: ast.AST):
        """Visits AST node and its children to replace template argument names with constant expressions."""

        # First, check if this node is correct wrt template arguments.
        self.check_node(node)

        for field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                if isinstance(field_value, ast.Name) and field_value.id in self.parameters:
                    new_node = ast.copy_location(self.parameters[field_value.id], field_value)
                    setattr(node, field_name, new_node)
                self.visit(field_value)

            # Node can be a list of statements.
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if isinstance(item, ast.AST):
                        if isinstance(item, ast.Name) and item.id in self.parameters:
                            field_value[i] = ast.copy_location(self.parameters[item.id], item)
                        self.visit(item)
