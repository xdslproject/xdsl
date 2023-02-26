from __future__ import annotations
from typing import Annotated
from xdsl.ir import Attribute, Dialect, OpResult, SSAValue
from xdsl.irdl import Operand, OpAttr, AnyAttr, Operation
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr


class Declare(Operation):
    name: str = "symref.declare"
    sym_name: OpAttr[StringAttr]

    @staticmethod
    def get(sym_name: str | StringAttr) -> Declare:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        return Declare.build(attributes={"sym_name": sym_name})


class Fetch(Operation):
    name: str = "symref.fetch"
    value: Annotated[OpResult, AnyAttr()]
    symbol: OpAttr[SymbolRefAttr]

    @staticmethod
    def get(symbol: str | SymbolRefAttr, result_type: Attribute) -> Fetch:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr.from_str(symbol)
        return Fetch.build(attributes={"symbol": symbol},
                           result_types=[result_type])


class Update(Operation):
    name: str = "symref.update"
    value: Annotated[Operand, AnyAttr()]
    symbol: OpAttr[SymbolRefAttr]

    @staticmethod
    def get(symbol: str | SymbolRefAttr,
            value: Operation | SSAValue) -> Update:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr.from_str(symbol)
        return Update.build(operands=[value], attributes={"symbol": symbol})


Symref = Dialect([Declare, Fetch, Update], [])
