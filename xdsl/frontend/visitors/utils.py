import ast
import importlib
import inspect

from dataclasses import dataclass
from typing import _GenericAlias, Any, Callable, Dict, List, Optional, Union
from xdsl.dialects.builtin import FloatData, IntAttr, StringAttr

from xdsl.ir import Attribute
from xdsl.irdl import VerifyException
import xdsl.frontend
import xdsl.frontend.region
import xdsl.frontend.dialects.builtin
from xdsl.frontend.internal_utils import is_frontend_obj, frontend_module_name_to_xdsl_name
from xdsl.frontend.visitors.type_to_literal import TypeToLiteralVisitor


@dataclass
class VisitorException(Exception):
    s: str

    def __str__(self) -> str:
        return f"Exception during translation of frontend AST to MLIR: {self.s}"


@dataclass
class UtilsException(Exception):
    s: str

    def __str__(self) -> str:
        return f"Exception in internal utils of frontend: {self.s}"


def _resolve(node: ast.AST, glob: Dict[str, Any]) -> Any:
    """
    Resolve a node in the context (globals()) of the frontend program.
    """
    return glob.get(ast.unparse(node), None)


def get_xdsl_obj(frontend_obj: Attribute, glob: Dict[str, Any],
                 args: Optional[List[Union[_GenericAlias, Attribute]]] = None) -> Union[_GenericAlias, Attribute]:
    """
    Get the xDSL class that corresponds to a class used in the frontend.
    """

    # Type translation needs to take into account the arguments too
    if isinstance(frontend_obj, _GenericAlias):
        return py_hint_to_xdsl_type(frontend_obj, glob)

    frontend_module = frontend_module_name_to_xdsl_name(
        frontend_obj.__module__)
    module = importlib.import_module(frontend_module)
    dialect_obj = module.__dict__[frontend_obj.__name__]

    if frontend_obj.__name__ not in module.__dict__:
        raise UtilsException(
            f"could not find translation for frontend object {frontend_obj}.")

    # The type hints of the frontend are fields of the class that represents
    # the base type in xDSL. The type is an instance of the base class with
    # instanciations of these fields.
    if args:
        return dialect_obj(args)
    elif hasattr(frontend_obj, "__args__"):
        dialect_obj = dialect_obj(frontend_obj.__args__)

    return dialect_obj


def py_hint_to_xdsl_type(stmt: Union[ast.Name, ast.Subscript, _GenericAlias],
                         glob: Dict[str, Any]) -> Union[_GenericAlias, Attribute]:
    """
    Convert a Python type hint from the frontend to an xDSL type (a Python class instance)
    """

    args = None
    if isinstance(stmt, ast.Name):
        type_name = stmt.id
    elif isinstance(stmt, ast.Subscript):
        args = py_hint_to_xdsl_type(stmt.slice, glob)
        if not hasattr(args, "__iter__"):
            args = [args]
        type_name = stmt.value.id
    elif isinstance(stmt, _GenericAlias):
        args = []
        for arg in stmt.__args__:
            if is_frontend_obj(arg):
                arg = get_xdsl_obj(arg, glob)

                # TODO: find a nicer way to translate aliases. The problem is f64
                #   in the frontend is an Alias for Float64Type. But in xDSL, it is an
                #   instance of Float64Type.
                if inspect.isclass(arg):
                    arg = arg()
            else:
                visitor = TypeToLiteralVisitor()
                arg = visitor.visit(arg)

                # TODO: handle default conversions more nicely (e.g. read from config file,
                #   don't hide them in this helper function but apply them more generally)
                match arg.__class__.__name__:
                    case "int":
                        arg = IntAttr.from_int(arg)
                    case "float":
                        arg = FloatData.from_float(arg)
                    case "str":
                        arg = StringAttr.from_str(arg)
                    case _:
                        raise VisitorException(f"NYI: default translation for literal {arg} "
                                               f"of type {type(arg)}) is not yet implemented.")

            print("\t-> transformed arg to", arg)
            args.append(arg)

        # The type hints of the frontend are fields of the class that represents
        # the base type in xDSL. The type is an instance of the base class with
        # instanciations of these fields.
        xdsl_class = get_xdsl_obj(stmt.__origin__, glob)

        try:
            return xdsl_class(args)
        except VerifyException as e:
            # Heuristicly try to add the type to the arguments
            typ = getattr(stmt.__origin__, "_default_typ", None)

            if typ:
                try:
                    return xdsl_class(args + [typ])
                except VisitorException:
                    pass

            raise VisitorException(f"Python type hint {stmt} could not be converted "
                                   f"because building '{xdsl_class}' failed with error:\n{e}")
    else:
        raise UtilsException(f"unexpected type, cannot convert {stmt} of "
                             f"type {type(stmt)} to xDSL type.")

    if type_name not in glob:
        raise UtilsException(f"invalid type name {type_name} is not "
                             "defined in the frontend program.")

    frontend_typ = glob[type_name]
    xdsl_obj = get_xdsl_obj(frontend_typ, glob, args)

    if inspect.isclass(xdsl_obj):
        # Classes without arguments are not instantiated in get_xdsl_obj
        return xdsl_obj()
    return xdsl_obj


def node_is_frontend_obj(node: ast.AST, glob: Dict[str, Any]) -> object:
    """
    Checks whether a node is a frontend object.
    :returns: frontend object or None.
    """
    return is_frontend_obj(_resolve(node, glob))


def get_xdsl_op(xdsl_obj_name: str, op_attr_name: str, glob: Dict[str, Any]) -> object:
    """
    Resolves the operation 'op_attr_name' for xdsl object 'xdsl_obj_name'.

    This uses the frontend program's context 'glob' to get the frontend object corresponding
    to 'xdsl_obj_name' and uses it to resolve 'op_attr_name' using the operation mapping defined
    by the frontend.

    :returns: xdsl operation for 'op_attr_name'
    """
    if xdsl_obj_name not in glob:
        raise UtilsException(
            f"Resolving {xdsl_obj_name} for operation '{op_attr_name}' failed.")

    obj = glob[xdsl_obj_name]
    if not hasattr(obj, op_attr_name):
        raise UtilsException(
            f"Value {obj} of type {type(obj)} has no implementation for operation '{op_attr_name}'.")

    # Call function with placeholder arguments, since they are anyway only used for static typing.
    # TODO: this needs to be changed when we find a nicer way to resolve operations.
    fn = getattr(obj, op_attr_name)
    args = [None] * len(inspect.signature(fn).parameters)
    return fn(*args)


def is_special_with_block(node: ast.With, glob: Dict[str, Any], special_class: object) -> bool:
    if len(node.items) != 1:
        return False

    item = node.items[0]
    if not isinstance(item.context_expr, ast.Call):
        return False

    return issubclass(_resolve(item.context_expr.func, glob), special_class)


def is_region(node: ast.With, glob: Dict[str, Any]) -> bool:
    return is_special_with_block(node, glob, xdsl.frontend.region.Region)


def is_module(node: ast.With, glob: Dict[str, Any]) -> bool:
    return is_special_with_block(node, glob, xdsl.frontend.dialects.builtin.Module)


def has_type_of_python_type(node: ast.Subscript | ast.Name | _GenericAlias):
    return type(node) in [ast.Subscript, ast.Name, _GenericAlias]
