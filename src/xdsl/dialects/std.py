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

        self.ctx.register_op(And)
        self.ctx.register_op(Or)
        self.ctx.register_op(Xor)

        # TODO move to builtin
        self.f32 = Float32Type()
        self.i64 = IntegerType.from_width(64)
        self.i32 = IntegerType.from_width(32)
        self.i1 = IntegerType.from_width(1)


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


@irdl_op_definition
class And(Operation):
    name: str = "std.and"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> And:
        return And.build(operands=[operand1, operand2],
                         result_types=[IntegerType.from_width(1)])


@irdl_op_definition
class Or(Operation):
    name: str = "std.or"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Or:
        return Or.build(operands=[operand1, operand2],
                        result_types=[IntegerType.from_width(1)])


@irdl_op_definition
class Xor(Operation):
    name: str = "std.xor"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Xor:
        return Xor.build(operands=[operand1, operand2],
                         result_types=[IntegerType.from_width(1)])
