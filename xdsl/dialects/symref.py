from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)


@irdl_op_definition
class DeclareOp(IRDLOperation):
    name = "symref.declare"
    sym_name = prop_def(StringAttr)

    assembly_format = "$sym_name attr-dict"

    def __init__(self, sym_name: str | StringAttr):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        super().__init__(properties={"sym_name": sym_name})


@irdl_op_definition
class FetchOp(IRDLOperation):
    name = "symref.fetch"
    value = result_def()
    symbol = prop_def(SymbolRefAttr)

    assembly_format = "$symbol attr-dict `:` type($value)"

    def __init__(self, symbol: str | SymbolRefAttr, result_type: Attribute):
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        super().__init__(properties={"symbol": symbol}, result_types=[result_type])


@irdl_op_definition
class UpdateOp(IRDLOperation):
    name = "symref.update"
    value = operand_def()
    symbol = prop_def(SymbolRefAttr)

    assembly_format = "$symbol `=` $value attr-dict `:` type($value)"

    def __init__(self, symbol: str | SymbolRefAttr, value: Operation | SSAValue):
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        super().__init__(operands=[value], properties={"symbol": symbol})


Symref = Dialect(
    "symref",
    [
        DeclareOp,
        FetchOp,
        UpdateOp,
    ],
    [],
)
