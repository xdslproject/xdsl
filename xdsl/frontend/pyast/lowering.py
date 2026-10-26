from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import xdsl.dialects.func as func
import xdsl.dialects.py as py
from xdsl.dialects.builtin import FileLineColLoc, FunctionType
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import Attribute, Block, BlockArgument, Operation, Region, SSAValue
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
