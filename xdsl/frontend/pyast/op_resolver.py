import ast
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass

from xdsl.frontend.pyast.dialects.builtin import (
    _FrontendType,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.frontend.pyast.exception import FrontendProgramException
from xdsl.ir import Operation


@dataclass
class OpResolver:
    """
    Class responsible to resolve frontend operations into xDSL operations.
    """

    @staticmethod
    def resolve_op(module_name: str, func_name: str) -> Callable[..., Operation]:
        module = importlib.import_module(module_name)
        resolver_name = "resolve_" + func_name
        if not hasattr(module, resolver_name):
            raise FrontendProgramException(
                f"Internal failure: operation '{func_name}' does not exist "
                f"in module '{module_name}'."
            )
        return getattr(module, resolver_name)()

    @staticmethod
    def resolve_op_overload(
        python_op: str, frontend_type: type[_FrontendType]
    ) -> Callable[..., Operation]:
        # First, get overloaded function.
        if not hasattr(frontend_type, python_op):
            raise FrontendProgramException(
                f"Internal failure: '{frontend_type.__name__}' does not "
                f"overload '{python_op}'."
            )
        overload = getattr(frontend_type, python_op)

        # Inspect overloaded function to extract what it maps to. By our
        # design, that should be a return call.
        #
        # def overload(...):
        #   from M import F
        #   return F(...)
        python_ast = ast.parse(inspect.getsource(overload).strip())
        func_ast = python_ast.body[0]
        assert isinstance(func_ast, ast.FunctionDef)

        if (
            len(func_ast.body) != 2
            or not isinstance(func_ast.body[0], ast.ImportFrom)
            or not isinstance(func_ast.body[1], ast.Return)
            or not isinstance(func_ast.body[1].value, ast.Call)
            or not isinstance(func_ast.body[1].value.func, ast.Name)
        ):
            msg = f"""
Internal failure while resolving '{python_op}'. Function AST for resolution is not correct, instead it should be:
    def __overload__(...):
        from Dialect import Operation
            return Operation(...)"""
            raise FrontendProgramException(msg)

        module_name = func_ast.body[0].module
        assert module_name is not None
        func_name = func_ast.body[1].value.func.id
        return OpResolver.resolve_op(module_name, func_name)
