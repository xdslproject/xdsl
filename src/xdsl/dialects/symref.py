from __future__ import annotations
from typing import List, Union
from dataclasses import dataclass

from xdsl.ir import Attribute, MLContext, SSAValue
from xdsl.irdl import (ResultDef, irdl_op_definition, AttributeDef, AnyAttr,
                       Operation, OperandDef)
from xdsl.dialects.builtin import (StringAttr, FlatSymbolRefAttr)


@dataclass
class Symref:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Declare)
        self.ctx.register_op(Update)
        self.ctx.register_op(Fetch)


@irdl_op_definition
class Declare(Operation):
    name: str = "symref.declare"

    sym_name = AttributeDef(StringAttr)

    @staticmethod
    def get(sym_name: StringAttr) -> Declare:
        return Declare.build(attributes={"sym_name": sym_name})


@irdl_op_definition
class Update(Operation):
    name: str = "symref.update"

    symbol = AttributeDef(FlatSymbolRefAttr)
    value = OperandDef(AnyAttr())

    @staticmethod
    def get(symbol: Union[str, FlatSymbolRefAttr], value: Operation | SSAValue) -> Update:
        return Update.build(operands=[value], attributes={"symbol": symbol})


@irdl_op_definition
class Fetch(Operation):
    name: str = "symref.fetch"

    symbol = AttributeDef(FlatSymbolRefAttr)
    value = ResultDef(AnyAttr())

    @staticmethod
    def get(symbol: Union[str, FlatSymbolRefAttr], result_type: Attribute) -> Fetch:
        return Fetch.build(attributes={"symbol": symbol}, result_types=[result_type])
