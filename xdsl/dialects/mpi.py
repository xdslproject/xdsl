from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, FunctionType, Attribute, FlatSymbolRefAttr
from xdsl.ir import MLContext, SSAValue
from xdsl.irdl import (OptAttributeDef, irdl_op_definition, VarOperandDef,
                       AnyAttr, Block, RegionDef, Region, Operation,
                       AttributeDef, VarResultDef)


@dataclass
class MpiCall:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(MPI_Send)


@irdl_op_definition
class MPI_Send(Operation):
    name: str = "func.MPI_Send"

    body = RegionDef()
    sym_name = AttributeDef(StringAttr)
    function_type = AttributeDef(FunctionType)
    sym_visibility = OptAttributeDef(StringAttr)

    @staticmethod
    def from_callable(name: str,
                      count: IntegerAttr,
                      input_types: List[Attribute],
                      return_types: List[Attribute],
                      func: Block.BlockCallback) -> MPI_Send:
        type_attr = FunctionType.from_lists(input_types, return_types)
        attributes = {
            "sym_name": name,
            "function_type": type_attr,
            "sym_visibility": "private"
        }
        op = MPI_Send.build(attributes=attributes,
                            regions=[Region.from_block_list(
                                     [Block.from_callable(input_types, func)])])
        return op
