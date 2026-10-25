from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import xdsl.dialects.func as func
import xdsl.dialects.py as py
import xdsl.dialects.symref as symref
from xdsl.dialects.builtin import FileLineColLoc, FunctionType
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.rewriter import InsertPoint, Rewriter
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

_BINARY_OPERATORS = {name.lower(): name for name in BINARY_OPERATOR_OVERLOADS}
_COMPARISON_OPERATORS = {
    name.lower(): name
    for name in (*COMPARISON_OPERATOR_OVERLOADS, *UNSUPPORTED_COMPARISON_OPERATORS)
}


@dataclass
class PyLowering:
    """
    Lowers the `py` operations generated from a Python function to the operations
    registered for them.

    Operations are lowered in order, so the types of the operands of an operation are
    known when it is lowered: every value gets the type of the operation it becomes.
    """

    type_converter: TypeConverter
    """The registries, and the globals in which names in the program are resolved."""

    file: str | None
    """Path of the file containing the program being processed."""

    function_definition_constructor: FunctionDefinitionConstructor = build_func
    """Construct the operation for each function."""

    return_constructor: ReturnConstructor = build_return
    """Construct return operations, optionally omitting the terminator."""

    variables: dict[str, Attribute] = field(
        default_factory=dict[str, Attribute], init=False
    )
    """The types of the variables of the function being lowered."""

    packs: dict[SSAValue, tuple[tuple[SSAValue, ...], Operation]] = field(
        default_factory=dict[SSAValue, tuple[tuple[SSAValue, ...], Operation]],
        init=False,
    )
    """
    The `py` values that stand for a number of values other than one, such as the
    results of a call, with the operation that produced them. These values cannot be
    used.
    """

    def lower(self, op: Operation) -> None:
        """Lower all `py` operations nested in `op`."""
        self._lower_regions(op)
        for _, producer in self.packs.values():
            Rewriter.erase_op(producer)
        self.packs.clear()

    def _lower_regions(self, op: Operation) -> None:
        for region in op.regions:
            for block in region.blocks:
                self._lower_block(block)

    def _lower_block(self, block: Block) -> None:
        for op in list(block.ops):
            self._lower_op(op)

    def _lower_op(self, op: Operation) -> None:
        for operand in op.operands:
            if operand in self.packs:
                raise self._error(
                    self.packs[operand][1],
                    "Expected an expression with exactly one result.",
                )

        match op:
            case py.PyFuncOp():
                self._lower_func(op)
            case py.PyReturnOp():
                self._lower_return(op)
            case py.PyCastOp():
                # Casts only change annotations, which are resolved by now.
                Rewriter.replace_op(op, (), (op.value,))
            case py.PyBinOp():
                self._lower_binop(op)
            case py.PyCompareOp():
                self._lower_compare(op)
            case py.PyCallOp():
                self._lower_call(op)
            case symref.FetchOp():
                name = op.symbol.root_reference.data
                Rewriter.replace_value_with_new_type(op.value, self.variables[name])
            case symref.UpdateOp():
                self._lower_update(op)
            case _:
                self._lower_regions(op)

    def _lower_func(self, op: py.PyFuncOp) -> None:
        block = op.body.block
        arg_types = [
            self._resolve_annotation(arg, arg_type, "argument")
            for arg, arg_type in zip(
                block.args, op.function_type.inputs.data, strict=True
            )
        ]
        (result_type,) = op.function_type.outputs.data
        return_types = (
            []
            if self._annotation(result_type) == "None"
            else [self._resolve_annotation(op, result_type, "return")]
        )

        for arg, arg_type in zip(block.args, arg_types, strict=True):
            location = arg.location
            Rewriter.replace_value_with_new_type(arg, arg_type)
            block.args[arg.index].location = location
        self.variables = {}
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

    def _lower_binop(self, op: py.PyBinOp) -> None:
        op_name = _BINARY_OPERATORS[op.op.data]

        # Check that the types of the operands are the same.
        # This is a (temporary?) restriction over Python for implementation simplicity.
        # This also means that we do not need to support reflected operations
        # (__radd__, __rsub__, etc.) which only exist for operations between different types.
        lhs, rhs = op.lhs, op.rhs
        if lhs.type != rhs.type:
            raise self._error(
                op,
                f"Expected the same types for binary operation '{op_name}', "
                f"but got {lhs.type} and {rhs.type}.",
            )

        source_type = self._source_type(op, lhs.type)
        overload = BINARY_OPERATOR_OVERLOADS[op_name]
        if (method_op := self._resolve_method(source_type, overload, lhs, rhs)) is None:
            raise self._error(
                op,
                f"Binary operation '{op_name}' "
                f"is not supported by type '{source_type.__qualname__}' "
                f"which does not overload '{overload}'.",
            )
        self._replace(op, method_op)

    def _lower_compare(self, op: py.PyCompareOp) -> None:
        op_name = _COMPARISON_OPERATORS[op.op.data]
        if op_name in UNSUPPORTED_COMPARISON_OPERATORS:
            raise self._error(op, f"Unsupported comparison operation '{op_name}'.")

        # Check that the types of the operands are the same.
        # This is a (temporary?) restriction over Python for implementation simplicity.
        # This also means that we do not need to consider swapping arguments
        # (__eq__ and __ne__ are their own reflection, __lt__ <-> __gt__  and __le__ <-> __ge__).
        lhs, rhs = op.lhs, op.rhs
        if lhs.type != rhs.type:
            raise self._error(
                op,
                f"Expected the same types for comparison operator '{op_name}',"
                f" but got {lhs.type} and {rhs.type}.",
            )

        source_type = self._source_type(op, lhs.type)
        overload = COMPARISON_OPERATOR_OVERLOADS[op_name]
        if (method_op := self._resolve_method(source_type, overload, lhs, rhs)) is None:
            raise self._error(
                op,
                f"Comparison operation '{op_name}' "
                f"is not supported by type '{cast(TypeAttribute, lhs.type).name}' "
                f"which does not overload '{overload}'.",
            )
        self._replace(op, method_op)

    def _lower_call(self, op: py.PyCallOp) -> None:
        callee = op.callee.data
        if "." in callee:
            kind = "classmethod"
            source = self._resolve_classmethod(op, *callee.split(".", maxsplit=1))
        else:
            kind = "function"
            source = self.type_converter.globals.get(callee, None)
            if source is None:
                raise self._error(op, f"Function '{callee}' is not defined in scope.")

        ir_op = self.type_converter.function_registry.get_operation_constructor(source)
        if ir_op is None:
            raise self._error(op, f"{kind.capitalize()} '{callee}' is not registered.")

        keywords = op.keyword_names
        num_positional = len(op.args) - len(keywords)
        args = op.args[:num_positional]
        kwargs = dict(zip(keywords, op.args[num_positional:], strict=True))
        self._replace(op, ir_op(*args, **kwargs))

    def _resolve_classmethod(
        self, op: py.PyCallOp, class_name: str, method_name: str
    ) -> Callable[..., Any]:
        source_class = self.type_converter.globals.get(class_name, None)
        if source_class is None:
            raise self._error(op, f"Class '{class_name}' is not defined in scope.")
        classmethod_ = getattr(source_class, method_name, None)
        if classmethod_ is None:
            raise self._error(
                op, f"Method '{method_name}' is not defined on class '{class_name}'."
            )
        return classmethod_

    def _lower_update(self, op: symref.UpdateOp) -> None:
        name = op.symbol.root_reference.data
        if name not in self.variables:
            self.variables[name] = op.value.type
        elif self.variables[name] != op.value.type:
            raise self._error(op, f"Cannot change the type of variable '{name}'.")

    def _replace(self, op: Operation, new_op: Operation) -> None:
        """Replace a `py` operation by `new_op`, which may have any number of results."""
        new_op.location = op.location
        if len(new_op.results) == 1:
            Rewriter.replace_op(op, new_op)
        else:
            Rewriter.insert_op(new_op, InsertPoint.before(op))
            self.packs[op.results[0]] = (tuple(new_op.results), op)

    def _source_type(self, op: Operation, ir_type: Attribute) -> type:
        source_type = self.type_converter.type_registry.get_annotation(
            cast(TypeAttribute, ir_type)
        )
        if source_type is None:
            raise self._error(
                op, f"IR type '{ir_type}' is not registered with a source type."
            )
        return source_type

    def _resolve_method(
        self, source_type: type, method_name: str, *args: SSAValue
    ) -> Operation | None:
        return self.type_converter.function_registry.resolve_operation(
            module_name=source_type.__module__,
            method_name=f"{source_type.__qualname__}.{method_name}",
            args=args,
        )

    @staticmethod
    def _annotation(object_type: Attribute) -> str:
        return cast(py.PyObjectType, object_type).annotation.data

    def _resolve_annotation(
        self, ir: Operation | BlockArgument, object_type: Attribute, kind: str
    ) -> Attribute:
        annotation = self._annotation(object_type)
        ir_type = self.type_converter.type_registry.resolve_attribute(
            annotation, self.type_converter.globals
        )
        if ir_type is None:
            raise self._error(ir, f"Unsupported function {kind} type: '{annotation}'")
        return ir_type

    @staticmethod
    def _position(ir: Operation | BlockArgument) -> tuple[int, int]:
        """The line and one-based column of an operation or argument, or zeros."""
        if isinstance(location := ir.location, FileLineColLoc):
            return location.line.data, location.column.data
        return 0, 0

    def _location(self, op: Operation) -> Location:
        return Location(self.file or "<unknown>", *self._position(op))

    def _error(
        self, ir: Operation | BlockArgument, msg: str
    ) -> CodeGenerationException:
        line, column = self._position(ir)
        # Errors report the zero-based column of the Python AST.
        return CodeGenerationException(self.file, line, max(column - 1, 0), msg)
