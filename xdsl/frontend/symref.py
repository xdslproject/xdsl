from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    Operand,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class Declare(IRDLOperation):
    name = "symref.declare"
    sym_name: StringAttr = attr_def(StringAttr)

    @staticmethod
    def get(sym_name: str | StringAttr) -> Declare:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        return Declare.build(attributes={"sym_name": sym_name})


@irdl_op_definition
class Fetch(IRDLOperation):
    name = "symref.fetch"
    value: OpResult = result_def(AnyAttr())
    symbol: SymbolRefAttr = attr_def(SymbolRefAttr)

    @staticmethod
    def get(symbol: str | SymbolRefAttr, result_type: Attribute) -> Fetch:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        return Fetch.build(attributes={"symbol": symbol}, result_types=[result_type])


@irdl_op_definition
class Update(IRDLOperation):
    name = "symref.update"
    value: Operand = operand_def(AnyAttr())
    symbol: SymbolRefAttr = attr_def(SymbolRefAttr)

    @staticmethod
    def get(symbol: str | SymbolRefAttr, value: Operation | SSAValue) -> Update:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        return Update.build(operands=[value], attributes={"symbol": symbol})


Symref = Dialect([Declare, Fetch, Update], [])
