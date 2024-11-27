from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class DeclareOp(IRDLOperation):
    name = "symref.declare"
    sym_name = attr_def(StringAttr)

    @staticmethod
    def get(sym_name: str | StringAttr) -> DeclareOp:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        return DeclareOp.build(attributes={"sym_name": sym_name})


@irdl_op_definition
class FetchOp(IRDLOperation):
    name = "symref.fetch"
    value = result_def()
    symbol = attr_def(SymbolRefAttr)

    @staticmethod
    def get(symbol: str | SymbolRefAttr, result_type: Attribute) -> FetchOp:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        return FetchOp.build(attributes={"symbol": symbol}, result_types=[result_type])


@irdl_op_definition
class UpdateOp(IRDLOperation):
    name = "symref.update"
    value = operand_def()
    symbol = attr_def(SymbolRefAttr)

    @staticmethod
    def get(symbol: str | SymbolRefAttr, value: Operation | SSAValue) -> UpdateOp:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        return UpdateOp.build(operands=[value], attributes={"symbol": symbol})


Symref = Dialect(
    "symref",
    [
        DeclareOp,
        FetchOp,
        UpdateOp,
    ],
    [],
)
