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

        self.f32 = Float32Type.get()
        self.i64 = IntegerType.get(64)
        self.i32 = IntegerType.get(32)
        self.i1 = IntegerType.get(1)

    # TODO make this generic in the type
    def constant(self, val: int, typ: Attribute) -> Operation:
        return Constant.create([], [typ],
                               attributes={"value": IntegerAttr.get(val, typ)})

    def constant_from_attr(self, attr: Attribute, typ: Attribute) -> Operation:
        return Constant.create([], [typ], attributes={"value": attr})

    def mulf(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Mulf.create([get_ssa_value(x), get_ssa_value(y)], [self.f32])

    def addf(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Addf.create([get_ssa_value(x), get_ssa_value(y)], [self.f32])

    def call(self, callee: str, ops: List[OpOrBlockArg],
             return_types: List[Attribute]) -> Operation:
        return Call.create(
            [get_ssa_value(op) for op in ops],
            return_types,
            attributes={"callee": FlatSymbolRefAttr.get(callee)})

    def return_(self, *ops: OpOrBlockArg) -> Operation:
        return Return.create([get_ssa_value(op) for op in ops], [])

    # TODO these operations should support all kinds of integer types
    def muli(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Muli.create([get_ssa_value(x), get_ssa_value(y)], [self.i32])

    def addi(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Addi.create([get_ssa_value(x), get_ssa_value(y)], [self.i32])

    def subi(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Subi.create([get_ssa_value(x), get_ssa_value(y)], [self.i32])

    def floordivi_signed(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return FloordiviSigned.create(
            [get_ssa_value(x), get_ssa_value(y)], [self.i32])

    def remi_signed(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return RemiSigned.create(
            [get_ssa_value(x), get_ssa_value(y)], [self.i32])

    # TODO these operations should support all kinds of integer types
    def and_(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return And.create([get_ssa_value(x), get_ssa_value(y)], [self.i1])

    def or_(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Or.create([get_ssa_value(x), get_ssa_value(y)], [self.i1])

    def xor_(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Xor.create([get_ssa_value(x), get_ssa_value(y)], [self.i1])

    def cmpi(self, x: OpOrBlockArg, y: OpOrBlockArg, arg: int) -> Operation:
        return Cmpi.create(
            [get_ssa_value(x), get_ssa_value(y)], [self.i1],
            attributes={"predicate": IntegerAttr.get(arg, self.i64)})


@irdl_op_definition
class Constant(Operation):
    name: str = "std.constant"
    output = ResultDef(AnyAttr())
    value = AttributeDef(AnyAttr())

    # TODO verify that the output and value type are equal
    def verify_(self) -> None:
        # TODO how to force the attr to have a type? and how to query it?
        pass


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


@irdl_op_definition
class Call(Operation):
    name: str = "std.call"
    arguments = VarOperandDef(AnyAttr())
    callee = AttributeDef(FlatSymbolRefAttr)

    # Note: naming this results triggers an ArgumentError
    res = VarResultDef(AnyAttr())
    # TODO how do we verify that the types are correct?


@irdl_op_definition
class Return(Operation):
    name: str = "std.return"
    arguments = VarOperandDef(AnyAttr())


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


@irdl_op_definition
class Cmpi(Operation):
    name: str = "std.cmpi"
    predicate = AttributeDef(IntegerAttr)
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType.get(1))


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
