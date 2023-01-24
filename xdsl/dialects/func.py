from __future__ import annotations
from typing import Annotated, List, Union

from xdsl.dialects.builtin import StringAttr, FunctionType, FlatSymbolRefAttr
from xdsl.ir import SSAValue, Operation, Block, Region, Attribute, Dialect
from xdsl.irdl import (VarOpResult, irdl_op_definition, VarOperand, AnyAttr,
                       OpAttr, OptOpAttr)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class FuncOp(Operation):
    name: str = "func.func"

    body: Region
    sym_name: OpAttr[StringAttr]
    function_type: OpAttr[FunctionType]
    sym_visibility: OptOpAttr[StringAttr]

    def verify_(self) -> None:
        # TODO: how to verify that there is a terminator?
        entry_block: Block = self.body.blocks[0]
        block_arg_types = [arg.typ for arg in entry_block.args]
        if self.function_type.inputs.data != block_arg_types:
            raise VerifyException(
                "Expected entry block arguments to have the same types as the function input types"
            )

    @staticmethod
    def from_callable(name: str, input_types: List[Attribute],
                      return_types: List[Attribute],
                      func: Block.BlockCallback) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": name,
            "function_type": type_attr,
            "sym_visibility": "private"
        }
        op = FuncOp.build(attributes=attributes,
                          regions=[
                              Region.from_block_list(
                                  [Block.from_callable(input_types, func)])
                          ])
        return op

    @staticmethod
    def from_region(name: str, input_types: List[Attribute],
                    return_types: List[Attribute], region: Region) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": name,
            "function_type": type_attr,
            "sym_visibility": "private"
        }
        op = FuncOp.build(attributes=attributes, regions=[region])
        return op


@irdl_op_definition
class Call(Operation):
    name: str = "func.call"
    arguments: Annotated[VarOperand, AnyAttr()]
    callee: OpAttr[FlatSymbolRefAttr]

    # Note: naming this results triggers an ArgumentError
    res: Annotated[VarOpResult, AnyAttr()]
    # TODO how do we verify that the types are correct?

    @staticmethod
    def get(callee: Union[str, FlatSymbolRefAttr],
            operands: List[Union[SSAValue, Operation]],
            return_types: List[Attribute]) -> Call:
        return Call.build(operands=[operands],
                          result_types=[return_types],
                          attributes={"callee": callee})


@irdl_op_definition
class Return(Operation):
    name: str = "func.return"
    arguments: Annotated[VarOperand, AnyAttr()]

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        function_return_types = func_op.function_type.outputs.data
        return_types = [arg.typ for arg in self.arguments]
        if function_return_types != return_types:
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )

    @staticmethod
    def get(*ops: Union[Operation, SSAValue]) -> Return:
        ops = [op for op in ops] if ops != () else []
        return Return.build(operands=[ops])


Func = Dialect([FuncOp, Call, Return], [])
