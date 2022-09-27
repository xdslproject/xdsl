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
        self.ctx.register_op(DivUI)
        self.ctx.register_op(DivSI)
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

        self.ctx.register_op(ShLI)
        self.ctx.register_op(Select)


@irdl_op_definition
class Constant(Operation):
    name: str = "arith.constant"
    result = ResultDef(AnyAttr())
    value = AttributeDef(AnyAttr())

    # TODO verify that the result and value type are equal

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
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addi:
        operand1 = SSAValue.get(operand1)
        return Addi.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Muli(Operation):
    name: str = "arith.muli"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Muli:
        operand1 = SSAValue.get(operand1)
        return Muli.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Subi(Operation):
    name: str = "arith.subi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Subi:
        operand1 = SSAValue.get(operand1)
        return Subi.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class DivUI(Operation):
    name: str = "arith.divui"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> DivUI:
        return DivUI.build(operands=[operand1, operand2],
                           result_types=[IntegerType.from_width(32)])


@irdl_op_definition
class DivSI(Operation):
    name: str = "arith.divsi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> DivSI:
        return DivSI.build(operands=[operand1, operand2],
                           result_types=[IntegerType.from_width(32)])


@irdl_op_definition
class FloorDivSI(Operation):
    name: str = "arith.floordivsi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> FloorDivSI:
        operand1 = SSAValue.get(operand1)
        return FloorDivSI.build(operands=[operand1, operand2],
                                result_types=[operand1.typ])


@irdl_op_definition
class CeilDivSI(Operation):
    name: str = "arith.ceildivsi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> CeilDivSI:
        operand1 = SSAValue.get(operand1)
        return CeilDivSI.build(operands=[operand1, operand2],
                               result_types=[operand1.typ])


@irdl_op_definition
class CeilDivUI(Operation):
    name: str = "arith.ceildivui"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> CeilDivUI:
        operand1 = SSAValue.get(operand1)
        return CeilDivUI.build(operands=[operand1, operand2],
                               result_types=[operand1.typ])


@irdl_op_definition
class RemSI(Operation):
    name: str = "arith.remsi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> RemSI:
        operand1 = SSAValue.get(operand1)
        return RemSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MinUI(Operation):
    name: str = "arith.minui"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MinUI:
        operand1 = SSAValue.get(operand1)
        return MinUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MaxUI(Operation):
    name: str = "arith.maxui"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MaxUI:
        operand1 = SSAValue.get(operand1)
        return MaxUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MinSI(Operation):
    name: str = "arith.minsi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MinSI:
        operand1 = SSAValue.get(operand1)
        return MinSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MaxSI(Operation):
    name: str = "arith.maxsi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MaxSI:
        operand1 = SSAValue.get(operand1)
        return MaxSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class AndI(Operation):
    name: str = "arith.andi"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> AndI:
        operand1 = SSAValue.get(operand1)
        return AndI.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class OrI(Operation):
    name: str = "arith.ori"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> OrI:
        operand1 = SSAValue.get(operand1)
        return OrI.build(operands=[operand1, operand2],
                         result_types=[operand1.typ])


@irdl_op_definition
class XOrI(Operation):
    name: str = "arith.xori"
    lhs = OperandDef(signlessIntegerLike)
    rhs = OperandDef(signlessIntegerLike)
    result = ResultDef(signlessIntegerLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

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
    lhs = OperandDef(IntegerType)
    rhs = OperandDef(IntegerType)
    result = ResultDef(IntegerType.from_width(1))

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
    lhs = OperandDef(floatingPointLike)
    rhs = OperandDef(floatingPointLike)
    result = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addf:
        operand1 = SSAValue.get(operand1)
        return Addf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Mulf(Operation):
    name: str = "arith.mulf"
    lhs = OperandDef(floatingPointLike)
    rhs = OperandDef(floatingPointLike)
    result = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Mulf:
        operand1 = SSAValue.get(operand1)
        return Mulf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Maxf(Operation):
    name: str = "arith.maxf"
    lhs = OperandDef(floatingPointLike)
    rhs = OperandDef(floatingPointLike)
    result = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Maxf:
        operand1 = SSAValue.get(operand1)
        return Maxf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Minf(Operation):
    name: str = "arith.minf"
    lhs = OperandDef(floatingPointLike)
    rhs = OperandDef(floatingPointLike)
    result = ResultDef(floatingPointLike)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Minf:
        operand1 = SSAValue.get(operand1)
        return Minf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Select(Operation):
    name: str = "arith.select"
    input1 = OperandDef(IntegerType.from_width(1))  # should be unsigned
    input2 = OperandDef(Attribute)
    input3 = OperandDef(Attribute)
    output = ResultDef(Attribute)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != IntegerType.from_width(1):
            raise VerifyException("Condition has to be of type !i1")
        if self.input2.typ != self.input3.typ or self.input3.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue], operand2: Union[Operation,
                                                                  SSAValue],
            operand3: Union[Operation, SSAValue], type: Attribute) -> Select:
        return Select.build(operands=[operand1, operand2, operand3],
                            result_types=[type])


@irdl_op_definition
class ShLI(Operation):
    """
    Integer left shift
    """
    name: str = "arith.shli"
    input1 = OperandDef(IntegerType)  # should be unsigned
    input2 = OperandDef(IntegerType)  # should be unsigned
    output = ResultDef(IntegerType)  # should be unsigned

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> ShLI:
        return ShLI.build(operands=[operand1, operand2],
                          result_types=[IntegerType.from_width(32)])