from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import IntegerType, Float32Type, IntegerAttr, FlatSymbolRefAttr


@dataclass
class Std:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Constant)

        self.ctx.register_op(Addi)
        self.ctx.register_op(Muli)
        self.ctx.register_op(Subi)
        self.ctx.register_op(FloordiviSigned)
        self.ctx.register_op(RemiSigned)

        self.ctx.register_op(Addf)
        self.ctx.register_op(Mulf)

        self.ctx.register_op(Call)
        self.ctx.register_op(Return)

        self.ctx.register_op(And)
        self.ctx.register_op(Or)
        self.ctx.register_op(Xor)

        self.ctx.register_op(Cmpi)

        self.f32 = Float32Type()
        self.i64 = IntegerType.from_width(64)
        self.i32 = IntegerType.from_width(32)
        self.i1 = IntegerType.from_width(1)


@irdl_op_definition
class Constant(Operation):
    name: str = "std.constant"
    output = ResultDef(AnyAttr())
    value = AttributeDef(AnyAttr())

    # TODO verify that the output and value type are equal

    @staticmethod
    def from_attr(attr: Attribute, typ: Attribute) -> Constant:
        return Constant.create(result_types=[typ], attributes={"value": attr})

    @staticmethod
    def from_int_constant(val: Union[int, Attribute],
                          typ: Union[int, Attribute]) -> Constant:
        if isinstance(typ, int):
            typ = IntegerType.from_width(typ)
        return Constant.create(
            result_types=[typ],
            attributes={"value": IntegerAttr.from_params(val, typ)})


@irdl_op_definition
class Addi(Operation):
    name: str = "std.addi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addi:
        return Addi.build(operands=[operand1, operand2],
                          result_types=[IntegerType.from_width(32)])


@irdl_op_definition
class Muli(Operation):
    name: str = "std.muli"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Muli:
        return Muli.build(operands=[operand1, operand2],
                          result_types=[IntegerType.from_width(32)])


@irdl_op_definition
class Subi(Operation):
    name: str = "std.subi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Subi:
        return Subi.build(operands=[operand1, operand2],
                          result_types=[IntegerType.from_width(32)])


@irdl_op_definition
class FloordiviSigned(Operation):
    name: str = "std.floordivi_signed"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> FloordiviSigned:
        return FloordiviSigned.build(operands=[operand1, operand2],
                                     result_types=[IntegerType.from_width(32)])


@irdl_op_definition
class RemiSigned(Operation):
    name: str = "std.remi_signed"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> RemiSigned:
        return RemiSigned.build(operands=[operand1, operand2],
                                result_types=[IntegerType.from_width(32)])


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


@irdl_op_definition
class Cmpi(Operation):
    name: str = "std.cmpi"
    predicate = AttributeDef(IntegerAttr)
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType.from_width(1))

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue], arg: int) -> Cmpi:
        return Cmpi.build(
            operands=[operand1, operand2],
            result_types=[IntegerType.from_width(1)],
            attributes={"predicate": IntegerAttr.from_int_and_width(arg, 64)})


@irdl_op_definition
class Addf(Operation):
    name: str = "std.addf"
    input1 = OperandDef(Float32Type)
    input2 = OperandDef(Float32Type)
    output = ResultDef(Float32Type)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addf:
        return Addf.build(operands=[operand1, operand2],
                          result_types=[Float32Type()])


@irdl_op_definition
class Mulf(Operation):
    name: str = "std.mulf"
    input1 = OperandDef(Float32Type)
    input2 = OperandDef(Float32Type)
    output = ResultDef(Float32Type)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Mulf:
        return Mulf.build(operands=[operand1, operand2],
                          result_types=[Float32Type()])
