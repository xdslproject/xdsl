from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import xdsl.dialects.func as func
import xdsl.dialects.py as py
from xdsl.dialects.builtin import FileLineColLoc, FunctionType
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue, TypeAttribute
from xdsl.rewriter import Rewriter
from xdsl.utils.lexer import Location

FunctionDefinitionConstructor = Callable[
    [str, FunctionType, Region, Location], Operation
]
"""Construct the operation for a function from its name, signature, body and location."""

ReturnConstructor = Callable[[Sequence[SSAValue], Location], Operation | None]
"""Construct a return operation, or return None when no terminator is needed."""


def build_func(
    name: str, signature: FunctionType, body: Region, location: Location
) -> func.FuncOp:
    return func.FuncOp(name, signature, body)


def build_return(values: Sequence[SSAValue], location: Location) -> func.ReturnOp:
    return func.ReturnOp(*values)


BINARY_OPERATOR_OVERLOADS = {
    "Add": "__add__",
    "Sub": "__sub__",
    "Mult": "__mul__",
    "Div": "__truediv__",
    "FloorDiv": "__floordiv__",
    "Mod": "__mod__",
    "Pow": "__pow__",
    "LShift": "__lshift__",
    "RShift": "__rshift__",
    "BitOr": "__or__",
    "BitXor": "__xor__",
    "BitAnd": "__and__",
    "MatMult": "__matmul__",
}
"""The method overloading each binary operator, by the name of its `ast` class."""

COMPARISON_OPERATOR_OVERLOADS = {
    "Eq": "__eq__",
    "Gt": "__gt__",
    "GtE": "__ge__",
    "Lt": "__lt__",
    "LtE": "__le__",
    "NotEq": "__ne__",
    "In": "__contains__",
}
"""The method overloading each comparison operator, by the name of its `ast` class."""

UNSUPPORTED_COMPARISON_OPERATORS = ("Is", "IsNot", "NotIn")
"""
The comparison operators without an overload: "is" and "is not" compare object
identity, which is not preserved, and "not in" is the negation of `__contains__`
rather than a method of its own.
"""

ErrorConstructor = Callable[[str], Exception]
"""Construct the error to raise from its message, located by the caller."""


def resolve_callee(
    type_converter: TypeConverter, callee: str, error: ErrorConstructor
) -> Callable[..., Operation]:
    """
    The constructor of the operation registered for `callee`, the name of a function
    or of a class method in the program's globals.
    """
    if "." in callee:
        kind = "classmethod"
        class_name, method_name = callee.split(".", maxsplit=1)
        source_class = type_converter.globals.get(class_name, None)
        if source_class is None:
            raise error(f"Class '{class_name}' is not defined in scope.")
        source = getattr(source_class, method_name, None)
        if source is None:
            raise error(
                f"Method '{method_name}' is not defined on class '{class_name}'."
            )
    else:
        kind = "function"
        source = type_converter.globals.get(callee, None)
        if source is None:
            raise error(f"Function '{callee}' is not defined in scope.")

    ir_op = type_converter.function_registry.get_operation_constructor(source)
    if ir_op is None:
        raise error(f"{kind.capitalize()} '{callee}' is not registered.")
    return ir_op


def resolve_binary_operator(
    type_converter: TypeConverter,
    op_name: str,
    lhs: SSAValue,
    rhs: SSAValue,
    error: ErrorConstructor,
) -> Operation:
    """The registered operation for the binary operator `op_name` on two values."""
    # Check that the types of the operands are the same.
    # This is a (temporary?) restriction over Python for implementation simplicity.
    # This also means that we do not need to support reflected operations
    # (__radd__, __rsub__, etc.) which only exist for operations between different types.
    if lhs.type != rhs.type:
        raise error(
            f"Expected the same types for binary operation '{op_name}', "
            f"but got {lhs.type} and {rhs.type}."
        )

    source_type = _source_type(type_converter, lhs.type, error)
    overload = BINARY_OPERATOR_OVERLOADS[op_name]
    op = _resolve_method(type_converter, source_type, overload, lhs, rhs)
    if op is None:
        raise error(
            f"Binary operation '{op_name}' "
            f"is not supported by type '{source_type.__qualname__}' "
            f"which does not overload '{overload}'."
        )
    return op


def resolve_comparison_operator(
    type_converter: TypeConverter,
    op_name: str,
    lhs: SSAValue,
    rhs: SSAValue,
    error: ErrorConstructor,
) -> Operation:
    """
    The registered operation for the comparison operator `op_name` on two values,
    which must not be one of `UNSUPPORTED_COMPARISON_OPERATORS`.
    """
    # Check that the types of the operands are the same.
    # This is a (temporary?) restriction over Python for implementation simplicity.
    # This also means that we do not need to consider swapping arguments
    # (__eq__ and __ne__ are their own reflection, __lt__ <-> __gt__  and __le__ <-> __ge__).
    if lhs.type != rhs.type:
        raise error(
            f"Expected the same types for comparison operator '{op_name}',"
            f" but got {lhs.type} and {rhs.type}."
        )

    source_type = _source_type(type_converter, lhs.type, error)
    overload = COMPARISON_OPERATOR_OVERLOADS[op_name]
    op = _resolve_method(type_converter, source_type, overload, lhs, rhs)
    if op is None:
        raise error(
            f"Comparison operation '{op_name}' "
            f"is not supported by type '{cast(TypeAttribute, lhs.type).name}' "
            f"which does not overload '{overload}'."
        )
    return op


def _source_type(
    type_converter: TypeConverter, ir_type: Attribute, error: ErrorConstructor
) -> type:
    source_type = type_converter.type_registry.get_annotation(
        cast(TypeAttribute, ir_type)
    )
    if source_type is None:
        raise error(f"IR type '{ir_type}' is not registered with a source type.")
    return source_type


def _resolve_method(
    type_converter: TypeConverter,
    source_type: type,
    method_name: str,
    *args: SSAValue,
) -> Operation | None:
    return type_converter.function_registry.resolve_operation(
        module_name=source_type.__module__,
        method_name=f"{source_type.__qualname__}.{method_name}",
        args=args,
    )


@dataclass
class PyLowering:
    """
    Lowers the `py` operations generated from a Python function to the operations
    registered for them.
    """

    type_converter: TypeConverter
    """The registries, and the globals in which names in the program are resolved."""

    file: str | None
    """Path of the file containing the program being processed."""

    function_definition_constructor: FunctionDefinitionConstructor = build_func
    """Construct the operation for each function."""

    return_constructor: ReturnConstructor = build_return
    """Construct return operations, optionally omitting the terminator."""

    def lower(self, op: Operation) -> None:
        """Lower all `py` operations nested in `op`."""
        self._lower_regions(op)

    def _lower_regions(self, op: Operation) -> None:
        for region in op.regions:
            for block in region.blocks:
                self._lower_block(block)

    def _lower_block(self, block: Block) -> None:
        for op in list(block.ops):
            self._lower_op(op)

    def _lower_op(self, op: Operation) -> None:
        match op:
            case py.PyFuncOp():
                self._lower_func(op)
            case py.PyReturnOp():
                self._lower_return(op)
            case py.PyCastOp():
                # Casts only change annotations, which are resolved by now.
                Rewriter.replace_op(op, (), (op.value,))
            case _:
                self._lower_regions(op)

    def _lower_func(self, op: py.PyFuncOp) -> None:
        function_type = op.function_type
        arg_types = [
            self._resolve_annotation(op, arg_type, "argument")
            for arg_type in function_type.inputs.data
        ]
        (result_type,) = function_type.outputs.data
        return_types = (
            []
            if self._annotation(result_type) == "None"
            else [self._resolve_annotation(op, result_type, "return")]
        )

        block = op.body.block
        for arg, arg_type in zip(block.args, arg_types, strict=True):
            Rewriter.replace_value_with_new_type(arg, arg_type)
        self._lower_block(block)

        function_op = self.function_definition_constructor(
            op.func_name.data,
            FunctionType.from_lists(arg_types, return_types),
            Rewriter.move_region_contents_to_new_regions(op.body),
            self._location(op),
        )
        Rewriter.replace_op(op, function_op, ())

    def _lower_return(self, op: py.PyReturnOp) -> None:
        value = op.value
        none = value.owner
        if not (isinstance(none, py.PyConstOp) and none.value is None):
            none = None

        # Returning None returns no values.
        values = () if none else (value,)
        return_op = self.return_constructor(values, self._location(op))
        if return_op is None:
            Rewriter.erase_op(op)
        else:
            Rewriter.replace_op(op, return_op, ())
        if none and not none.res.uses:
            Rewriter.erase_op(none)

    @staticmethod
    def _annotation(object_type: Attribute) -> str:
        return cast(py.PyObjectType, object_type).annotation.data

    def _resolve_annotation(
        self, op: Operation, object_type: Attribute, kind: str
    ) -> Attribute:
        annotation = self._annotation(object_type)
        ir_type = self.type_converter.type_registry.resolve_attribute(
            annotation, self.type_converter.globals
        )
        if ir_type is None:
            raise self._error(op, f"Unsupported function {kind} type: '{annotation}'")
        return ir_type

    @staticmethod
    def _position(op: Operation) -> tuple[int, int]:
        """The line and one-based column of an operation, or zeros."""
        if isinstance(location := op.location, FileLineColLoc):
            return location.line.data, location.column.data
        return 0, 0

    def _location(self, op: Operation) -> Location:
        return Location(self.file or "<unknown>", *self._position(op))

    def _error(self, op: Operation, msg: str) -> CodeGenerationException:
        line, column = self._position(op)
        # Errors report the zero-based column of the Python AST.
        return CodeGenerationException(self.file, line, max(column - 1, 0), msg)
