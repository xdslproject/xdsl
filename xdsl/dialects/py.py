"""
The py dialect is intended to represent a Python-like language.

It is high-level enough to be translated back into Python.
However, it also includes additional operators to represent more complex mechanisms, such as exceptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xdsl.dialects.builtin import FunctionType, StringAttr, SymbolNameConstraint
from xdsl.dialects.func import CallOpSymbolUserOpInterface
from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Dialect,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    GenericData,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    CallableOpInterface,
    ConstantLike,
    HasParent,
    IsTerminator,
    Pure,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class Type(GenericData[str], TypeAttribute):
    name = "py.type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data)

    @staticmethod
    def constr() -> AttrConstraint:
        return AnyAttr()


@irdl_attr_definition
class ConstantValue(Data[Any]):
    name = "py.value"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.__str__())

    @staticmethod
    def constr() -> AttrConstraint:
        return AnyAttr()


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "py.constant"

    value = prop_def(ConstantValue)
    result = result_def(Type)

    traits = traits_def(
        ConstantLike(),
        Pure(),
    )

    def __init__(self, value: Any, result_type: Type):
        super().__init__(properties={"value": value}, result_types=[result_type])


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body

    @classmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "py.func"

    body = region_def("single_block")
    sym_name = prop_def(SymbolNameConstraint())
    function_type = attr_def(FunctionType)

    traits = traits_def(SymbolOpInterface(), FuncOpCallableInterface())

    def __init__(
        self,
        name: str,
        arg_names: list[str],
        ftype: FunctionType,
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        properties: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": ftype,
        }
        if not isinstance(region, Region):
            region = Region(Block(arg_types=ftype.inputs))

        if region.first_block is None:
            region.add_block(Block(arg_types=ftype.inputs))

        assert region.first_block is not None

        for i in range(len(arg_names)):
            region.first_block.args[i].name_hint = arg_names[i]

        super().__init__(properties=properties, regions=[region])

    def verify_(self):
        block = self.body.block

        if not block.ops:
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.last_op

        if last_op is None or not last_op.has_trait(IsTerminator):
            raise VerifyException("Expected last op of FuncOp to be a ReturnOp")


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "py.return"
    input = opt_operand_def(Type)
    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, input: SSAValue | None = None):
        return super().__init__(operands=[input])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        function_return_type = func_op.function_type.outputs.data
        if function_return_type != tuple(self.operand_types):
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    Converts the input type into an output type
    """

    name = "py.cast"

    input = operand_def()
    result_type = result_def()

    traits = traits_def(
        # Pure(), # ?
    )

    def __init__(self, input: SSAValue, result_type: Type):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class AssertOp(IRDLOperation):
    name = "py.cast"

    input = operand_def()
    result_type = result_def()

    traits = traits_def(
        # Pure(), # ?
    )

    def __init__(self, input: SSAValue, result_type: Type):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "py.call"

    callee = prop_def()
    arguments = var_operand_def()
    res = var_result_def()

    traits = traits_def(CallOpSymbolUserOpInterface())

    def __init__(
        self,
        callee: str | StringAttr,
        arguments: Sequence[SSAValue | Operation],
        return_type: Sequence[Attribute],
    ):
        if isinstance(callee, str):
            callee = StringAttr(callee)

        super().__init__(
            operands=[arguments],
            result_types=return_type,
            properties={"callee": callee},
        )


@irdl_op_definition
class PassOp(IRDLOperation):
    name = "py.pass"

    traits = traits_def(IsTerminator())

    def __init__(self):
        super().__init__()


Py = Dialect(
    "py",
    [
        ConstantOp,
        FuncOp,
        ReturnOp,
        CastOp,
        CallOp,
        PassOp,
    ],
    [
        Type,
        ConstantValue,
    ],
)
