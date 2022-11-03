import ast

from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.codegen.type_conversion import TypeHintConverter


def check_function_signature(node: ast.FunctionDef):
    """Throws an exception if this function cannot be lowered to xDSL."""
    # Don't support vararg and its friends.
    if getattr(node.args, "vararg") is not None:
        raise CodegenException("`vararg` arguments are not supported")
    if getattr(node.args, "kwarg") is not None:
        raise CodegenException("`kwarg` arguments are not supported")
    if getattr(node.args, "kwonlyargs"):
        raise CodegenException("`kwonlyargs` are not supported")
    if getattr(node.args, "kw_defaults"):
        raise CodegenException("`kw_defaults` are not supported")
    if getattr(node.args, "defaults"):
        raise CodegenException("`defaults` are not supported")

    # Explicitly require type annotations on function arguments.
    args = node.args.args
    for i, arg in enumerate(args):
        annotation = arg.annotation
        if annotation is None:
            # TODO: Compiler should complain about all arguments which miss type
            # annotations, and not just the first one.
            raise CodegenException(f"missing a type hint on argument {i} in \
                                     function {node.name}, line {annotation.lineno}.")


def get_argument_types(node: ast.FunctionDef, converter: TypeHintConverter):
    """Converts argument types to xDSL types."""
    args = node.args.args
    arg_types = []
    for arg in args:
        arg_type = converter.convert_hint(arg.annotation)
        arg_types.append(arg_type)
    return arg_types


def get_return_types(node: ast.FunctionDef, converter: TypeHintConverter):
    """Converts return types to xDSL types."""
    return_type = converter.convert_hint(node.returns)
    return_types = []
    if return_type is not None:
        return_types.append(return_type)
    return return_types
