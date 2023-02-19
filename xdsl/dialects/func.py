from __future__ import annotations
from typing import Annotated, List, Union

from xdsl.dialects.builtin import StringAttr, FunctionType, SymbolRefAttr
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
                "Expected entry block arguments to have the same types as the function "
                "input types")

    @staticmethod
    def from_callable(name: str, input_types: List[Attribute],
                      return_types: List[Attribute],
                      func: Block.BlockCallback) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": StringAttr(name),
            "function_type": type_attr,
            "sym_visibility": StringAttr("private")
        }
        op = FuncOp.build(attributes=attributes,
                          regions=[
                              Region.from_block_list(
                                  [Block.from_callable(input_types, func)])
                          ])
        return op

    @staticmethod
    def external(name: str, input_types: List[Attribute],
                 return_types: List[Attribute]) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": StringAttr(name),
            "function_type": type_attr,
            "sym_visibility": StringAttr("private")
        }
        op = FuncOp.build(attributes=attributes,
                          regions=[Region.from_operation_list([])])
        return op

    @staticmethod
    def from_region(name: str, input_types: List[Attribute],
                    return_types: List[Attribute], region: Region) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": StringAttr(name),
            "function_type": type_attr,
            "sym_visibility": StringAttr("private")
        }
        op = FuncOp.build(attributes=attributes, regions=[region])
        return op


@irdl_op_definition
class Call(Operation):
    name: str = "func.call"
    arguments: Annotated[VarOperand, AnyAttr()]
    callee: OpAttr[SymbolRefAttr]

    # Note: naming this results triggers an ArgumentError
    res: Annotated[VarOpResult, AnyAttr()]
    # TODO how do we verify that the types are correct?

    @staticmethod
    def get(callee: Union[str, SymbolRefAttr], ops: List[Union[SSAValue,
                                                               Operation]],
            return_types: List[Attribute]) -> Call:
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        return Call.build(operands=[ops],
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
    def get(*ops: Operation | SSAValue) -> Return:
        return Return.build(operands=[list(ops)])


Func = Dialect([FuncOp, Call, Return], [])
