from __future__ import annotations
from typing import Annotated
from xdsl.ir import Attribute, Dialect, OpResult, SSAValue
from xdsl.irdl import Operand, irdl_op_definition, OpAttr, AnyAttr, Operation
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr


@irdl_op_definition
class Declare(Operation):
    name: str = "symref.declare"
    sym_name: OpAttr[StringAttr]

    @staticmethod
    def get(sym_name: str | StringAttr) -> Declare:
        return Declare.build(attributes={"sym_name": sym_name})


@irdl_op_definition
class Fetch(Operation):
    name: str = "symref.fetch"
    value: Annotated[OpResult, AnyAttr()]
    symbol: OpAttr[SymbolRefAttr]

    @staticmethod
    def get(symbol: str | SymbolRefAttr, result_type: Attribute) -> Fetch:
        return Fetch.build(attributes={"symbol": symbol},
                           result_types=[result_type])


@irdl_op_definition
class Update(Operation):
    name: str = "symref.update"
    value: Annotated[Operand, AnyAttr()]
    symbol: OpAttr[SymbolRefAttr]

    @staticmethod
    def get(symbol: str | SymbolRefAttr,
            value: Operation | SSAValue) -> Update:
        return Update.build(operands=[value], attributes={"symbol": symbol})


Symref = Dialect([Declare, Fetch, Update], [])
