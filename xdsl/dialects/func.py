from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union

from xdsl.dialects.builtin import StringAttr, FunctionType, FlatSymbolRefAttr
from xdsl.ir import MLContext, SSAValue, Operation, Block, Region, Attribute
from xdsl.irdl import (OptAttributeDef, irdl_op_definition, VarOperandDef,
                       AnyAttr, RegionDef, AttributeDef, VarResultDef)


@dataclass
class Func:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(FuncOp)
        self.ctx.register_op(Call)
        self.ctx.register_op(Return)


@irdl_op_definition
class FuncOp(Operation):
    name: str = "func.func"

    body = RegionDef()
    sym_name = AttributeDef(StringAttr)
    function_type = AttributeDef(FunctionType)
    sym_visibility = OptAttributeDef(StringAttr)

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
    arguments = VarOperandDef(AnyAttr())
    callee = AttributeDef(FlatSymbolRefAttr)

    # Note: naming this results triggers an ArgumentError
    res = VarResultDef(AnyAttr())
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
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*ops: Union[Operation, SSAValue]) -> Return:
        ops = [op for op in ops] if ops != () else []
        return Return.build(operands=[ops])
