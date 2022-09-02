from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from xdsl.dialects.builtin import ContainerOf, Float16Type, Float64Type, IndexType, IntegerType, Float32Type, IntegerAttr
from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.irdl import (AnyOf, irdl_op_definition, AttributeDef, AnyAttr,
                       ResultDef, OperandDef, VerifyException, Attribute)

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))


@dataclass
class Arith:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Constant)

        self.ctx.register_op(Addi)
        self.ctx.register_op(Muli)
        self.ctx.register_op(Subi)
        self.ctx.register_op(FloorDivSI)
        self.ctx.register_op(CeilDivSI)
        self.ctx.register_op(CeilDivUI)
        self.ctx.register_op(RemSI)
        self.ctx.register_op(MinSI)
        self.ctx.register_op(MaxSI)
        self.ctx.register_op(MinUI)
        self.ctx.register_op(MaxUI)

        self.ctx.register_op(Addf)
        self.ctx.register_op(Mulf)

        self.ctx.register_op(Cmpi)

        self.ctx.register_op(AndI)
        self.ctx.register_op(OrI)
        self.ctx.register_op(XOrI)
        self.ctx.register_op(Minf)
        self.ctx.register_op(Maxf)


@irdl_op_definition
class Constant(Operation):
    name: str = "arith.constant"
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
    name: str = "arith.addi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addi:
        operand1 = SSAValue.get(operand1)
        return Addi.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Muli(Operation):
    name: str = "arith.muli"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Muli:
        operand1 = SSAValue.get(operand1)
        return Muli.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Subi(Operation):
    name: str = "arith.subi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Subi:
        operand1 = SSAValue.get(operand1)
        return Subi.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class FloorDivSI(Operation):
    name: str = "arith.floordivsi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> FloorDivSI:
        operand1 = SSAValue.get(operand1)
        return FloorDivSI.build(operands=[operand1, operand2],
                                result_types=[operand1.typ])


@irdl_op_definition
class CeilDivSI(Operation):
    name: str = "arith.ceildivsi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> CeilDivSI:
        operand1 = SSAValue.get(operand1)
        return CeilDivSI.build(operands=[operand1, operand2],
                               result_types=[operand1.typ])


@irdl_op_definition
class CeilDivUI(Operation):
    name: str = "arith.ceildivui"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> CeilDivUI:
        operand1 = SSAValue.get(operand1)
        return CeilDivUI.build(operands=[operand1, operand2],
                               result_types=[operand1.typ])


@irdl_op_definition
class RemSI(Operation):
    name: str = "arith.remsi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> RemSI:
        operand1 = SSAValue.get(operand1)
        return RemSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MinUI(Operation):
    name: str = "arith.minui"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MinUI:
        operand1 = SSAValue.get(operand1)
        return MinUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MaxUI(Operation):
    name: str = "arith.maxui"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MaxUI:
        operand1 = SSAValue.get(operand1)
        return MaxUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MinSI(Operation):
    name: str = "arith.minsi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MinSI:
        operand1 = SSAValue.get(operand1)
        return MinSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MaxSI(Operation):
    name: str = "arith.maxsi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MaxSI:
        operand1 = SSAValue.get(operand1)
        return MaxSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class AndI(Operation):
    name: str = "arith.andi"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> AndI:
        operand1 = SSAValue.get(operand1)
        return AndI.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class OrI(Operation):
    name: str = "arith.ori"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> OrI:
        operand1 = SSAValue.get(operand1)
        return OrI.build(operands=[operand1, operand2],
                         result_types=[operand1.typ])


@irdl_op_definition
class XOrI(Operation):
    name: str = "arith.xori"
    input1 = OperandDef(signlessIntegerLike)
    input2 = OperandDef(signlessIntegerLike)
    output = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> XOrI:
        operand1 = SSAValue.get(operand1)
        return XOrI.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Cmpi(Operation):
    name: str = "arith.cmpi"
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
    name: str = "arith.addf"
    input1 = OperandDef(floatingPointLike)
    input2 = OperandDef(floatingPointLike)
    output = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addf:
        operand1 = SSAValue.get(operand1)
        return Addf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Mulf(Operation):
    name: str = "arith.mulf"
    input1 = OperandDef(floatingPointLike)
    input2 = OperandDef(floatingPointLike)
    output = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Mulf:
        operand1 = SSAValue.get(operand1)
        return Mulf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Maxf(Operation):
    name: str = "arith.maxf"
    input1 = OperandDef(floatingPointLike)
    input2 = OperandDef(floatingPointLike)
    output = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Maxf:
        operand1 = SSAValue.get(operand1)
        return Maxf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Minf(Operation):
    name: str = "arith.minf"
    input1 = OperandDef(floatingPointLike)
    input2 = OperandDef(floatingPointLike)
    output = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Minf:
        operand1 = SSAValue.get(operand1)
        return Minf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])
