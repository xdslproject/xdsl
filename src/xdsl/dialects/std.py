from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import IntegerType, Float32Type, IntegerAttr, FlatSymbolRefAttr


@dataclass
class Std:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Call)
        self.ctx.register_op(Return)


@irdl_op_definition
class Call(Operation):
    name: str = "std.call"
    arguments = VarOperandDef(AnyAttr())
    callee = AttributeDef(FlatSymbolRefAttr)

    # Note: naming this results triggers an ArgumentError
    res = VarResultDef(AnyAttr())
    # TODO how do we verify that the types are correct?

    @staticmethod
    def get(callee: Union[str, FlatSymbolRefAttr],
            operands: List[Union[SSAValue, Operation]],
            return_types: List[Attribute]) -> Call:
        return Call.build(operands=operands,
                          result_types=return_types,
                          attributes={"callee": callee})


@irdl_op_definition
class Return(Operation):
    name: str = "std.return"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*ops: Union[Operation, SSAValue]) -> Return:
        return Return.build(operands=[[op for op in ops]])
