import ast
import importlib
import inspect

from dataclasses import dataclass
import xdsl.frontend.dialects.arith as src_arith
import xdsl.dialects.arith as dst_arith
from xdsl.frontend.dialects.builtin import FrontendType

from xdsl.ir import Operation


# Maps operator names to Python function which a type can override.
op_to_python = {
    "Add": "__add__",
    "BitAnd": "__and__",
    "Mult": "__mul__",
    "RShift": "__rshift__",
    "LShift": "__lshift__",
    "Sub": "__sub__",
    "Eq": "__eq__",
    "NotEq": "__ne__",
    "LtE": "__le__",
    "Lt": "__lt__",
    "Gt": "__gt__",
    "GtE": "__ge__",
    "Pow": "__pow__",

    # These do not really have a name.
    "__getitem__": "__getitem__",
    "__setitem__": "__setitem__",
}


@dataclass
class ResolverException(Exception):
    """Exception type for resolver failures."""
    msg: str

    def __str__(self) -> str:
        return f"Exception in resolver: {self.msg}."


@dataclass
class OpResolver:
    """Class responsible to resolve frontend operations into xDSL operations."""

    @staticmethod
    def resolve_method(module_name: str, func_name: str) -> Operation:
        module = importlib.import_module(module_name)
        resolver_name = "resolve_" + func_name
        if hasattr(module, resolver_name):
            resolver = getattr(module, resolver_name)
            return resolver
        return None

    @staticmethod
    def resolve_op_overload(op_name: str, frontend_type: FrontendType) -> Operation:
        if op_name not in op_to_python:
            raise ResolverException(f"unknown operator '{op_name}'")
        
        # First, get overloaded function.
        python_func_name = op_to_python[op_name]
        if not hasattr(frontend_type , python_func_name):
            raise ResolverException(f"{frontend_type.__name__} does not overload '{python_func_name}'")
            
        overload = getattr(frontend_type, python_func_name)

        # Inspect overloaded function to extract what it maps to. By our
        # design, that should be a return call.

        python_ast = ast.parse(inspect.getsource(overload).strip())
        if isinstance(python_ast, ast.Module) and isinstance(python_ast.body[0], ast.FunctionDef):
            func_ast = python_ast.body[0]

            # For now, the convention is to overload like:
            #
            # def overload(...):
            #   from M import F
            #   return F(...)
            if len(func_ast.body) != 2 or not isinstance(func_ast.body[0], ast.ImportFrom) or not isinstance(func_ast.body[1], ast.Return):
                return None
            return_value = func_ast.body[1].value
            if not isinstance(return_value, ast.Call) or not isinstance(return_value.func, ast.Name):
                return None

            module_name = func_ast.body[0].module
            func_name = return_value.func.id
            return OpResolver.resolve_method(module_name, func_name)
        
        # Otherwise resolution fails.
        return None
